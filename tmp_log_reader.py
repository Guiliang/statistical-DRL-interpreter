import pickle

with open("./interface/tmp_sub_action_score.pkl", 'rb') as f:
    scores = pickle.load(f)

print('Working')