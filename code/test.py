import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from DeepHotResi import *

# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/model_focal/"



def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                y_true = Variable(labels.cuda())

            else:
                node_features = Variable(node_features.float())
                adj_matrix = Variable(adj_matrix)
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()

            y_pred = model(node_features, G_batch, adj_matrix)

    

            # calculate loss
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict




def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold is None:
        best_auc = 0
        best_threshold = 0
        for threshold in range(0, 100):  # 步长为0.001
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            auc = metrics.roc_auc_score(y_true, binary_pred)
            if auc > best_auc:
                best_auc = auc
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, binary_pred).ravel()
    

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0


    binary_acc = metrics.accuracy_score(y_true, binary_pred)
    precision = metrics.precision_score(y_true, binary_pred)
    recall = metrics.recall_score(y_true, binary_pred) 
    f1 = metrics.f1_score(y_true, binary_pred)
    AUC = metrics.roc_auc_score(y_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(y_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall, # SEN
        'specificity': specificity, # SPE
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def gets(labels, list_pos):
    seq_len = [len(i) for i in labels]
    print(seq_len)
    left = [0]
    temp = 0
    for i in range(0, len(seq_len)):
        temp += seq_len[i]
        left.append(temp)
    print(left)
    data = []
    for i in range(len(list_pos)):
        for j in list_pos[i]:
            data.append(j + left[i])
    print('data',data)
    return data


def test(test_dataframe, psepos_path):
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe,psepos_path=psepos_path), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=graph_collate)
    dataset=ProDataset(dataframe=test_dataframe,psepos_path=psepos_path)
    print(dataset.names)

    _1FEU_A = [86, 9, 18, 84, 13, 19, 15] 
    _1WNE_A = [19] 
    _1ZDI_A = [84, 56, 60, 48, 42, 51]
    _2KXN_B = [90, 89, 93, 5, 88]
    _2XB2_G = [11, 5]
    _3AM1_A = [193]
    _3VYY_A = [29]
    _3UZS_A = [280] 
    _4CIO_A = [11, 53, 19, 85, 78, 88, 81, 83, 49, 51]
    _4G0A_A = [221]
    _4JVH_A = [155, 157, 87, 90, 64, 95]
    _4NL3_D = [5, 42]
    _5EN1_A = [54, 37, 12, 96]
    _5EV1_A = [147, 221, 245]
    _5HO4_A = [51, 77, 34, 9, 4, 93]


    all_data = [ _1FEU_A, _1WNE_A, _1ZDI_A, _2KXN_B, _2XB2_G, _3AM1_A, _3UZS_A, _3VYY_A,  _4CIO_A, _4G0A_A, _4JVH_A, _4NL3_D, _5EN1_A, _5EV1_A, _5HO4_A]
    
    index = gets(dataset.labels, all_data)
    # index = [i for i in range(300, 1500)]

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = DeepHotResi(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name))
        model.eval()
        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)
        
        print(len(test_pred))
        test_pred = [test_pred[i] for i in index]
        test_true = [test_true[i] for i in index]
        print('test_pred',test_pred)
        print('test_true',test_true)

        result_test = analysis(test_true, test_pred)

    print("========== Evaluate Test set ==========")
    print("Test binary acc: ", result_test['binary_acc'])
    print("Test precision:", result_test['precision'])
    print("Test recall (SEN): ", result_test['recall'])  
    print("Test specificity (SPE): ", result_test['specificity'])  
    print("Test f1: ", result_test['f1'])
    print("Test AUC: ", result_test['AUC'])
    print("Test AUPRC: ", result_test['AUPRC'])
    print("Test mcc: ", result_test['mcc'])


      

def test_one_dataset(dataset, psepos_path):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe, psepos_path)


def main():
    with open(Dataset_Path + "data_dict_test.pkl", "rb") as f:
        Test = pickle.load(f)

    Test_psepos_Path = './Dataset/protein_dict_test.pkl'

    print("Evaluate GraphPPIS on Test_60")
    test_one_dataset(Test, Test_psepos_Path)


if __name__ == "__main__":
    main()
