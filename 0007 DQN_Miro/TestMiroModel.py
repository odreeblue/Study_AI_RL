import torch
import numpy as np
PATH = "FindMiro.pt"
model = torch.load(PATH)
print(model)
model.eval()

state = [1,0,0,0,0,0,0,0,0] # 초기 위치
state = np.array(state) # 추가한 부분
state = torch.from_numpy(state).type(torch.FloatTensor)
state = torch.unsqueeze(state, 0)

action = model(state).max(1)[1].view(1,1)
print(action)

state = [0,0,0,1,0,0,0,0,0] # 초기 위치
state = np.array(state) # 추가한 부분
state = torch.from_numpy(state).type(torch.FloatTensor)
state = torch.unsqueeze(state, 0)
action = model(state).max(1)[1].view(1,1)
print(action)

state = [0,0,0,0,1,0,0,0,0] # 초기 위치
state = np.array(state) # 추가한 부분
state = torch.from_numpy(state).type(torch.FloatTensor)
state = torch.unsqueeze(state, 0)
action = model(state).max(1)[1].view(1,1)
print(action)

state = [0,0,0,0,0,0,0,1,0] # 초기 위치
state = np.array(state) # 추가한 부분
state = torch.from_numpy(state).type(torch.FloatTensor)
state = torch.unsqueeze(state, 0)
action = model(state).max(1)[1].view(1,1)
print(action)

state = [0,0,0,0,0,0,0,0,1] # 초기 위치
state = np.array(state) # 추가한 부분
state = torch.from_numpy(state).type(torch.FloatTensor)
state = torch.unsqueeze(state, 0)
action = model(state)
print(action)

