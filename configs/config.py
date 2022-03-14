
window_size = 20
topK = 256

# MS1M feat_dim=256, CASIA and VGG feat_dim=512
feat_dim = 256
dataset = 'MS1M'
test_name = 'part1_test'

knn_path = './data/{}/knns/{}_faiss_top{}.npz'.format(dataset, test_name, topK)
feat_path = './data/{}/features/{}.bin'.format(dataset, test_name)
label_path = './data/{}/labels/{}.meta'.format(dataset, test_name)
result_path = './result/{}/part1_test_top{}_winds{}.npy'.format(dataset, topK, window_size)
