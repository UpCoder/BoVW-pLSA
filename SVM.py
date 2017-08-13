from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
def do_svm(data, label, n_flod = 3):
    kf = KFold(n_splits=n_flod, shuffle=True)
    max_accuracy = 0.0
    record = []
    for param_c in range(-20, 20, 1):
        for param_g in range(-20, 20, 1):
            svc = SVC(
                C=pow(2, param_c),
                gamma=pow(2, param_g)
            )
            accuracy = 0.0
            for train_index, test_index in kf.split(data, label):
                svc.fit(data[train_index, :], label[train_index])
                predicted_label = svc.predict(data[test_index, :])
                ground_true = label[test_index]
                accuracy += accuracy_score(ground_true, predicted_label)
            accuracy /= n_flod
            if accuracy > max_accuracy:
                record = []
                record.append(param_c)
                record.append(param_g)
                max_accuracy = accuracy
            print '-c param is %g, -g params is %g, accuracy is %g' % (param_c, param_g, accuracy)
    print 'max accuracy is %g, -c params is %g, -g params is %g' % (max_accuracy, record[0], record[1])
    return max_accuracy, record