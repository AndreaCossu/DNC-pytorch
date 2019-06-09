'''
Created on Jul 18, 2018

@author: andrea
'''

import argparse
import torch
import torch.optim as optim
import time
from core.DNC import DNC
from CopyTask.train import train, get_dataset, masked_BCE_with_logits, get_dataset2, train2
from CopyTask.test import test, accuracy, accuracy2, test2

millis = int(round(time.time() * 1000))

parser = argparse.ArgumentParser()
parser.add_argument('epochs', type=int)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--N', type=int, default=64)
parser.add_argument('--W', type=int, default=8)
parser.add_argument('--R', type=int, default=2)
parser.add_argument('--vector_len', type=int, default=4)
parser.add_argument('--min_length_train', type=int, default=2)  # min sequence length to copy during training
parser.add_argument('--max_length_train', type=int, default=5)  # max sequence length to copy during training
parser.add_argument('--min_length_test', type=int, default=7)  # min sequence length to copy during validation and testing
parser.add_argument('--max_length_test', type=int, default=10)  # max sequence length to copy during validaiton and testing
parser.add_argument('--momentum', type=float, default=0.7)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--mlp_layers', type=int, default=0) # set it > 0 to choose a MLP controller
parser.add_argument('--lstm_layers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--cuda', action="store_true")
parser.add_argument('--load', action="store_true")
parser.add_argument('--no_save', action="store_true")
parser.add_argument('--print_every', type=int, default=1000)
args = parser.parse_args()

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight,std=0.2)


torch.manual_seed(millis)
mode = 'cpu'
if args.cuda:
    if torch.cuda.is_available():
        print('Using ', torch.cuda.device_count() ,' GPU(s)')
        mode = 'cuda'
    else:
        print("WARNING: No GPU found. Using CPUs...")
else:
    print('Using 0 GPUs')

device = torch.device(mode)
controller_type = 'MLP' if args.mlp_layers > 0 else 'LSTM'

if controller_type == 'MLP':
    num_layers = args.mlp_layers
else:
    num_layers = args.lstm_layers

path = "models/DNC.pt"
num_min_vectors = args.min_length_train
num_max_vectors = args.max_length_train
test_num_min_vectors = args.min_length_test
test_num_max_vectors = args.max_length_test


inputSize = args.vector_len
#outputSize = args.vector_len
outputSize = args.vector_len - 1


#output_f = torch.sigmoid
output_f = None
dnc = DNC(inputSize,args.hidden_size, num_layers, outputSize,args.N,args.W,args.R,
          args.batch_size, device, controller_type, output_f)
dnc.apply(init_weights)

if args.load:
    dnc.load_model(path)

optimizer = optim.RMSprop(dnc.parameters(), lr=args.learning_rate, momentum=args.momentum, eps=1e-10, weight_decay=0.)
#criterion = masked_BCE_with_logits
criterion = torch.nn.BCEWithLogitsLoss()



#inputsVal, targetsVal, valMasks = get_dataset(args.vector_len, test_num_min_vectors, test_num_max_vectors, args.batch_size, device)
inputsVal, targetsVal = get_dataset2(args.vector_len, test_num_min_vectors, test_num_max_vectors, args.batch_size, device)

avg_loss = 0.
best_val_loss = 1000.
for i in range(args.epochs):
    #inputs, targets, masks = get_dataset(args.vector_len, num_min_vectors, num_max_vectors, args.batch_size, device)
    #avg_loss += train(dnc, inputs, targets, masks, criterion, optimizer, device)

    inputs, targets = get_dataset2(args.vector_len, num_min_vectors, num_max_vectors, args.batch_size, device)
    avg_loss += train2(dnc, inputs, targets, criterion, optimizer, device)


    if ((i+1) % args.print_every) == 0:
        #valOut, valLoss = test(dnc, inputsVal, device, targetsVal, criterion, valMasks)
        #acc = accuracy(valOut, targetsVal, valMasks)
        valOut, valLoss = test2(dnc, inputsVal, device, targetsVal, criterion)
        acc = accuracy2(valOut, targetsVal)

        # save best model on validation set
        if not args.no_save:
            if best_val_loss > valLoss:
                best_val_loss = valLoss
                dnc.save_model(path)
        print()
        print("Epoch ", i+1, ": training loss = ", avg_loss / float(args.print_every))
        print("Epoch", i+1, ": validation loss =", valLoss)
        print("Epoch", i+1, ": validation accuracy =", acc, "%")
        avg_loss = 0.


#testInputs, testTargets, testMasks = get_dataset(args.vector_len, test_num_min_vectors, test_num_max_vectors, args.batch_size, device)
testInputs, testTargets = get_dataset2(args.vector_len, test_num_min_vectors, test_num_max_vectors, args.batch_size, device)

#outs, loss = test(dnc, testInputs, device, testTargets, criterion, testMasks)
#acc = accuracy(outs, testTargets, testMasks)

outs, loss = test2(dnc, testInputs, device, testTargets, criterion)
acc = accuracy2(outs, testTargets)

print("Loss on test set: ", loss)
print("Accuracy on test set: ", acc, "%")
print("Result ( 1 -> correct, 0 -> wrong):")
print(outs[0].round() == testTargets[0])
