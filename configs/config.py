
window_size = 20
topK = 256
feat_dim = 256
dataset = 'MS1M'
knn_path = '../data/{}/knns/part1_test_faiss_top{}.npz'.format(dataset, topK)
feat_path = '../data/{}/features/part1_test.bin'.format(dataset)
label_path = '../data/{}/labels/part1_test.meta'.format(dataset)
result_path = '../result/{}/part1_test_top{}_winds{}.npy'.format(dataset, topK, window_size)
