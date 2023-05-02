import argparse
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tqdm import tqdm

from cluster_dataloader import clusterDataset2 
from model import Encoder, Decoder, Seq2SeqClusters
from load_data import ClusterDatasetVocab

SOS = 0
EOS = 1
PAD = 2

def char2id(x):
    if type(x) == int:
        return x
    ord_num = ord(x)
    # +3 to create space for SOS, EOS, PAD
    return ord(x) - ord('a') + 3


def train(model, args, device, vocab):

    dataset = clusterDataset2(vocab, device=device, transform=torch.tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    #ctc_loss = nn.CTCLoss(blank=100)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        losses = []
        epoch_loss = 0
        for batch_idx, (data, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            y = model(data, targets)
            targets = targets
            y = y.permute(1, 2, 0)
            y = F.log_softmax(y, dim=1)
            loss = criterion(y, targets)
            #log_probs =  log_probs.transpose(0,1)
            #input_lengths = torch.full(
            #    size=(args.batch_size,), fill_value=log_probs.size(0), dtype=torch.int32
            #)
            #target_lengths = torch.full(
            #    size=(args.batch_size,), fill_value=targets.size(1), dtype=torch.int32
            #)
            #loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)


            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("===========================================")
                print(torch.argmax(y, dim=1))
                print(targets)
                print("Loss: ", sum(losses)/len(losses))
                print("===========================================")
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"/scratch/work/sarvasm1/av_hubert/ussee/cluster_id_to_char/exp/lstm_checkpoint.pt")

        print('Epoch {}: {}'.format(epoch, sum(losses)/len(losses)))

        torch.save(model.state_dict(), "/scratch/work/sarvasm1/av_hubert/ussee/cluster_id_to_char/exp/lstm.pt")

#def decode(model, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    parser.add_argument('--clusters', type=str)
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    input_vocab_size = 2004
    output_vocab_size = 29

    #INPUT_DIM = len(SRC.vocab)
    #OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256

    #vocab_from_pretrained = None
    vocab_from_pretrained = "/scratch/work/sarvasm1/av_hubert/ussee/cluster_id_to_char/exp/vocab.pt"

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")

    if vocab_from_pretrained is not None:
        vocab = torch.load(vocab_from_pretrained)
    else:
        vocab = ClusterDatasetVocab(args.text, args.clusters)
        torch.save(vocab, "/scratch/work/sarvasm1/av_hubert/ussee/cluster_id_to_char/exp/vocab.pt")

    encoder = Encoder(input_vocab_size, ENC_EMB_DIM, HID_DIM, n_layers=1)
    decoder = Decoder(vocab.tgt_vocab_size, DEC_EMB_DIM, HID_DIM, n_layers=1)

    model = Seq2SeqClusters(encoder, decoder, device).to(device)

    train(model, args, device, vocab)



