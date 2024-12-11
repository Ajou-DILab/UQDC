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
                #logits = logits[0]
                #alpha = F.relu(logits) + 1
                uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
                _, preds = torch.max(alpha, 1)
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                probs += prob.tolist()
                uncertainties += uncertainty.tolist()
                predss += preds.tolist()
                correct += (true_label == preds).sum().cpu()
                p = torch.sigmoid(logits)
                b_out = bce(p.detach().cpu(), true_label.detach().cpu())
                #print(b_out)
                ece_loss_list.append(b_out)
                true_labels += true_label.tolist()

        #y_true = true_label 
        #correct = sum(1 for a, b in zip(y_true, predss) if a == b) 
        #acc = correct / len(predss) 
    return probs, uncertainties, predss, correct / num_samples, true_labels
