import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Seq2SeqClusters(nn.Module):
    def __init__(self, encoder, decoder, device, sos=0, eos=1):
        super(Seq2SeqClusters, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.softmax = nn.Softmax(dim=2)
        self.device = device

        self.sos = sos
        self.eos = eos

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        #breakpoint()
        input = torch.tensor([0], device=self.device)

        # first input to the decoder is the <sos> token.
        #input = trg[0, :]
        for t in range(0, trg_len):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output

            ## decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            ## get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            ## update input : use ground_truth when teacher_force 
            input = trg[:, t] if teacher_force else top1
            #input = output.argmax(1)
        #outputs = self.softmax(outputs)
        return outputs

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(Encoder, self).__init__()

        self.h_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        embeddings = self.dropout(self.embed(src))
        output, (h, c) = self.rnn(embeddings)

        return h, c

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_vocab_size
        self.embed_dim = embedding_dim
        self.h_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.1)
        self.fc= nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, h, c):
        input = input.unsqueeze(0)
        embeddings = self.dropout(self.embed(input))
        output, (h, c) = self.rnn(embeddings, (h,c))
        prediction = self.fc(output.squeeze(0))

        return prediction, h, c
