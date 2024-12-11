def test_pred(net, device, dataloader, num_samples, with_labels=True):
    net.eval()
    probs = []
    uncertainties = []
    predss = []
    ece_loss_list = []
    true_labels = []
    correct = 0 
    with torch.no_grad():
        if with_labels:
            for q_ids, q_mask, q_token, label in tqdm(dataloader):
                q_ids, q_mask, q_token, true_label = q_ids.to(device), q_mask.to(device), q_token.to(device), label.to(device)
                logits, alpha = net(q_ids, q_mask, q_token, true_label)

                _, preds = torch.max(alpha, 1)


                predss += preds.tolist()
                correct += (true_label == preds).sum().cpu()
                p = torch.sigmoid(logits)
                probs += p.tolist()
                #print(b_out)
                true_labels += true_label.tolist()

        #y_true = true_label 
        #correct = sum(1 for a, b in zip(y_true, predss) if a == b) 
        #acc = correct / len(predss) 
    return predss, correct / num_samples, probs
