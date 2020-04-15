# import pickle
#
# with open("./interface/tmp_sub_action_score.pkl", 'rb') as f:
#     scores = pickle.load(f)
#
# print('Working')
import optparse
import sys


optparser = optparse.OptionParser()
optparser.add_option("-g", "--max_node_hist", dest="G", default=3000,
                     help="max number of instance in every node (default = 10000)")
opts = optparser.parse_args()[0]
print(opts.G)
sys.stderr.write('abc')

