from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
import weka.core.serialization as serialization
import weka.core.jvm as jvm
import csv

def generate_weka_training_data(data, action_id, dir):

    with open(dir, mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',')
        title = []
        [title.append('dim{0}'.format(i)) for  i in range(len(data[0][0]))]
        title.append('Advantage')
        employee_writer.writerow(title)
        for data_line in data:
            if data_line[1] == action_id:
                employee_writer.writerow(data_line[0].tolist()+[data_line[-1]])

class M5Tree():
    def __init__(self, model_name):
        self.model_name = model_name
        self.classifier = None
        jvm.start()


    def train_weka_model(self, training_data_dir, save_model_dir, log_file, mimic_env=None):
        """
        Just runs some example code.
        """
        loader = Loader(classname="weka.core.converters.CSVLoader")
        training_data = loader.load_file(training_data_dir)
        training_data.class_is_last()


        if self.model_name == 'm5-rt':  # m5 regression tree
            options = ["-R", "-M", "10"]
        elif self.model_name == 'm5-mt':  # m5 model tree
            options = ["-M", "10"]
        else:
            raise ValueError("unknown model name {0}".format(self.model_name))

        # classifier help, check https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/M5P.html
        self.classifier = Classifier(classname="weka.classifiers.trees.M5P", options=options)
        self.classifier.build_classifier(training_data)
        # print(classifier)
        graph = self.classifier.graph
        node_number = float(graph.split('\n')[-3].split()[0].replace('N', ''))
        leave_number = node_number / 2
        serialization.write(save_model_dir, self.classifier)
        print('leaves number of is {0}'.format(leave_number), file=log_file)

        evaluation = Evaluation(training_data)
        predicts = evaluation.test_model(self.classifier, training_data)

        if mimic_env is not None:
            predict_dictionary = {}
            for predict_index in range(len(predicts)):
                predict_value = predicts[predict_index]
                if predict_value in predict_dictionary.keys():
                    predict_dictionary[predict_value].append(predict_index)
                else:
                    predict_dictionary.update({predict_value:[predict_index]})

            return_value = mimic_env.get_return(state=list(predict_dictionary.values()))
            print(return_value, file=log_file)

        summary = evaluation.summary()
        numbers = summary.split('\n')
        corr = float(numbers[1].split()[-1])
        mae = float(numbers[2].split()[-1])
        rmse = float(numbers[3].split()[-1])
        rae = float(numbers[4].split()[-2]) / 100
        rrse = float(numbers[5].split()[-2]) / 100
        # print(evl)
        print(summary, file=log_file)
        jvm.stop()

        return corr, mae, rmse, rae, rrse


    def test_weka_model(self, testing_data_dir, save_model_dir, log_file, mimic_env=None):
        classifier = Classifier(jobject=serialization.read(save_model_dir))

        loader = Loader(classname="weka.core.converters.CSVLoader")
        testing_data = loader.load_file(testing_data_dir)
        testing_data.class_is_last()

        evaluation = Evaluation(testing_data)
        predicts = evaluation.test_model(classifier, testing_data)

        if mimic_env is not None:
            predict_dictionary = {}
            for predict_index in range(len(predicts)):
                predict_value = predicts[predict_index]
                if predict_value in predict_dictionary.keys():
                    predict_dictionary[predict_value].append(predict_index)
                else:
                    predict_dictionary.update({predict_value:[predict_index]})

            return_value = mimic_env.get_return(state=list(predict_dictionary.values()))
            print(return_value, file=log_file)

        summary = evaluation.summary()
        numbers = summary.split('\n')
        corr = float(numbers[1].split()[-1])
        mae = float(numbers[2].split()[-1])
        rmse = float(numbers[3].split()[-1])
        rae = float(numbers[4].split()[-2]) / 100
        rrse = float(numbers[5].split()[-2]) / 100
        # print(evl)
        print(summary, file=log_file)
        jvm.stop()

        return corr, mae, rmse, rae, rrse


if __name__ == "__main__":
    m5t = M5Tree (model_name='m5-rt')
    m5t.train_weka_model(training_data_dir="./tmp_m5.csv", save_model_dir="tmp.m5p.weka.model")