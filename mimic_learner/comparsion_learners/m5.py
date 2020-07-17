from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
import weka.core.serialization as serialization
import weka.core.jvm as jvm
import csv

def generate_weka_training_data(data, action_id, dir):

    with open(dir, mode='w') as csv_file:
        employee_writer = csv.writer(csv_file, delimiter=',')
        title = []
        [title.append('dim{0}'.format(i)) for  i in range(len(data[0][0]))]
        title.append('Advantage')
        employee_writer.writerow(title)
        for data_line in data:
            if data_line[1] == action_id:
                employee_writer.writerow(data_line[0].tolist()+[data_line[-1]])

class M5Tree():
    def __init__(self, model_name, options):
        self.model_name = model_name
        self.classifier = None
        self.options = options
        jvm.start()


    def train_weka_model(self, training_data_dir, save_model_dir, log_file, mimic_env=None):
        """
        Just runs some example code.
        """
        loader = Loader(classname="weka.core.converters.CSVLoader")
        training_data = loader.load_file(training_data_dir)
        training_data.class_is_last()

        self.classifier = Classifier(classname="weka.classifiers.trees.M5P", options=self.options)
        # classifier help, check https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/M5P.html
        self.classifier.build_classifier(training_data)
        # print(classifier)
        graph = self.classifier.graph
        node_number = float(graph.split('\n')[-3].split()[0].replace('N', ''))
        leaves_number = node_number / 2
        serialization.write(save_model_dir, self.classifier)
        # print('Leaves number is {0}'.format(leave_number), file=log_file)

        evaluation = Evaluation(training_data)
        predicts = evaluation.test_model(self.classifier, training_data)
        return_value = None
        if mimic_env is not None:
            predict_dictionary = {}
            for predict_index in range(len(predicts)):
                predict_value = predicts[predict_index]
                if predict_value in predict_dictionary.keys():
                    predict_dictionary[predict_value].append(predict_index)
                else:
                    predict_dictionary.update({predict_value:[predict_index]})

            return_value = mimic_env.get_return(state=list(predict_dictionary.values()))
            # print("Training return is {0}".format(return_value), file=log_file)

        summary = evaluation.summary()
        numbers = summary.split('\n')
        corr = float(numbers[1].split()[-1])
        mae = float(numbers[2].split()[-1])
        rmse = float(numbers[3].split()[-1])
        rae = float(numbers[4].split()[-2]) / 100
        rrse = float(numbers[5].split()[-2]) / 100
        # print(evl)
        # print("Training summary is "+summary, file=log_file)

        return return_value, mae, rmse, leaves_number


    def test_weka_model(self, testing_data_dir, save_model_dir, log_file, mimic_env=None):
        self.classifier = Classifier(jobject=serialization.read(save_model_dir))

        graph = self.classifier.graph
        node_number = float(graph.split('\n')[-3].split()[0].replace('N', ''))
        leaves_number = node_number / 2
        serialization.write(save_model_dir, self.classifier)
        # print('Leaves number is {0}'.format(leave_number), file=log_file)

        loader = Loader(classname="weka.core.converters.CSVLoader")
        testing_data = loader.load_file(testing_data_dir)
        testing_data.class_is_last()

        evaluation = Evaluation(testing_data)
        predicts = evaluation.test_model(self.classifier, testing_data)
        return_value = None
        if mimic_env is not None:
            predict_dictionary = {}
            for predict_index in range(len(predicts)):
                predict_value = predicts[predict_index]
                if predict_value in predict_dictionary.keys():
                    predict_dictionary[predict_value].append(predict_index)
                else:
                    predict_dictionary.update({predict_value:[predict_index]})

            return_value = mimic_env.get_return(state=list(predict_dictionary.values()))
            # print("Testing return is {0}".format(return_value), file=log_file)

        summary = evaluation.summary()
        numbers = summary.split('\n')
        corr = float(numbers[1].split()[-1])
        mae = float(numbers[2].split()[-1])
        rmse = float(numbers[3].split()[-1])
        rae = float(numbers[4].split()[-2]) / 100
        rrse = float(numbers[5].split()[-2]) / 100
        # print(evl)
        # print("Testing summary is "+summary, file=log_file)

        return return_value, mae, rmse, leaves_number



    def __del__(self):
        jvm.stop()

if __name__ == "__main__":
    m5t = M5Tree (model_name='m5-rt')
    m5t.train_weka_model(training_data_dir="./tmp_m5.csv", save_model_dir="tmp.m5p.weka.model")