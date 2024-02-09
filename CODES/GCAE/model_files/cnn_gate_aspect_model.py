import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.args = args
        # Number of elements in vocabulary
        V = args.embed_num
        # Dimensionality
        D = args.embed_dim
        # Number of features
        C = args.class_num

        # Number of aspects
        A = args.aspect_num

        # Number of kernels
        Co = args.kernel_num
        # Kernel size
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

        # y = Wx + b operation
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(args.aspect_embed_dim, Co)

    def forward(self, feature, aspect):
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.aspect_embed(aspect)  # (N, L', D)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        # concatinating tensors of given dimension (dim=1 in this case)
        x0 = torch.cat(x0, 1)
        logit = self.fc1(x0)  # (N,C)
        return logit, x, y
