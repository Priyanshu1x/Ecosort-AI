import torch
import torch.nn as nn
def train_model(model,dataloader1,dataloader2,lr,epochs,checkpoint_path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    best_val_accuracy=0
    fulldata={
        'train_losses':[],
        'train_accuracies':[],
        'valid_losses':[],
        'valid_accuracies':[]

    }
    for epoch in range(epochs):
        train_loss=0
        valid_loss=0
        train_correct=0
        valid_correct=0
        train_total=0
        valid_total=0
        model.train()
        for batch_feature,batch_label in dataloader1:
            batch_feature,batch_label=batch_feature.to(device),batch_label.to(device)
            model.zero_grad()
            y_pred=model(batch_feature)
            loss=criterion(y_pred,batch_label)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
            prediction=torch.argmax(y_pred,dim=1)
            train_correct+=(prediction==batch_label).sum().item()
            train_total+=batch_label.size(0)

        avg_train_loss=train_loss/len(dataloader1)
        fulldata['train_losses'].append(avg_train_loss)
        avg_train_accuracy=train_correct/train_total
        fulldata['train_accuracies'].append(avg_train_accuracy)


            
        model.eval()
        with torch.no_grad():
            for batch_feature,batch_label in dataloader2:
                batch_feature,batch_label=batch_feature.to(device),batch_label.to(device)    
                y_pred=model(batch_feature)
                loss=criterion(y_pred,batch_label)
                valid_loss+=loss.item()
                prediction=torch.argmax(y_pred,dim=1)
                valid_correct+=(prediction==batch_label).sum().item()
                valid_total+=batch_label.size(0)

        avg_val_loss=valid_loss/len(dataloader2)
        fulldata['valid_losses'].append(avg_val_loss)
        avg_val_accuracy=valid_correct/valid_total
        fulldata['valid_accuracies'].append(avg_val_accuracy)


        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy=avg_val_accuracy
            torch.save(model.state_dict(),checkpoint_path)
            print("best model saved---->")


        print(f"train loss: {avg_train_loss}| train accuracy: {avg_train_accuracy} | valid loss: {avg_val_loss} | valid accuracy: {avg_val_accuracy} ")
    state_dict=torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    return model,fulldata
        

        

        



