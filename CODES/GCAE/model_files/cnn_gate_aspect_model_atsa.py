import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard way of declaring a neural network
class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.args = args
        
        # Number of words in vocabulary
        V = args.embed_num
        # Dimension
        D = args.embed_dim
        
        # Number of classes
        C = args.class_num
        # Number of aspects
        A = args.aspect_num

        # Number of kernels
        Co = args.kernel_num
        # Size of kernel
        Ks = args.kernel_sizes

        # Creating a V X D matrix containing the semantical relations
        self.embed = nn.Embedding(V, D)
        # Creating parameter list associated with the model
        self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        # Creating the embedding matrix for aspects
        self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        # Creating parameter list associated with the model
        self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)

        # Creating a module list of 1d convolutions applied on kernels
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        # A convolutional layer is added to the design for ACSA
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]])


        # self.convs3 = nn.Conv1d(D, 300, 3, padding=1), smaller is better
        # makes some of the elements in input tensor zero with 0.2 probability
        self.dropout = nn.Dropout(0.2)

        # y = Wx + b operation
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(100, Co)


    def forward(self, feature, aspect):
        # Looking at the features of a particular element in vocabulary?
        feature = self.embed(feature)  # (N, L, D)
        # Looking at features of a particular entry in aspects?
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        # aa = F.tanhshrink(self.convs3(aspect_v.transpose(1, 2)))  # [(N,Co,L), ...]*len(Ks)
        # aa = F.max_pool1d(aa, aa.size(2)).squeeze(2)
        # aspect_v = aa
        # smaller is better

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]

        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # x = [F.adaptive_max_pool1d(i, 2) for i in x]
        # x = [i.view(i.size(0), -1) for i in x]

        # concatinating tensors of given dimension (dim=1 in this case)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        return logit, x, y
