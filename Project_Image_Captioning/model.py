import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        
        # image shape : [batch, channel, width, height]
        features = self.resnet(images)
        # featrue shape after resnet : [1(batch_size), 2048, 1, 1]
   
        features = features.view(features.size(0), -1)
        # feature shape after view: [1(batch_size), 2048]
        
        features = self.embed(features)
        # feature shape after embed: [1, 512]
        
        
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedded = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.score = nn.Linear(hidden_size, vocab_size)
        
    
    def init_hidden(self):
        reuturn (torch.randn(self.num_layers, self.batch_size, self.hidden_size), torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        
    def forward(self, features, captions):
        features = features.view(len(features),1, -1)
        # now feature shape : [batch_size, 1, embed_size]
             
        # caption shape : [batch_size, caption_length] including <end>
        # caption[:,:,-1] shape : [batch_size, captions_length-1] excluding <end>
        embeddings = self.embedded(captions[:,:-1])
        # embedding shape : [batch_size, caption_length-1, embed_size]
        
        inputs = torch.cat((features, embeddings),1)
        # inputs shape : [batch_size, captions_length, embed_size]
        
        out, hidden = self.lstm(inputs)
        # output shape after lstm: [batch_size, captions_length, hidden_size]
        out = self.score(out)
        # output shape after score layer: [batch_size, captions_length, vocab_size]
        return out
        
    def sample(self, inputs, states=None, max_len=20):
        """ accepts pre-processed image tensor (inputs) and 
            returns predicted sentence (list of tensor ids of length max_len) 
        """
        output_list=[]
        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            # output shape : [batch, 1, 512]
            
            output = self.score(output.squeeze(1))
            # output after score layer : [batch, vocab]
            
            # output.max(1) consists of score and vocab index
            # thus max_index is most likely vocab index
            max_index = output.max(1)[1]
            # max_index is still tensor. Use .item() to get int
            output_list.append(max_index.item())
            
            inputs = self.embedded(max_index)
            # input shape : [batch, 512]
               
            inputs = inputs.unsqueeze(1)
            # now input shape : [batch, 1, 512]
            
        return output_list