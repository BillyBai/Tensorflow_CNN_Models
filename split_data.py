import os
import random
import shutil
from config import train_data_path, test_data_path, train_list_path, test_list_path

root_path = "./data/GarbageData"
train_rate = 0.9
test_rate = 1 - train_rate


def main():
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    train_data_txt = open(train_list_path, 'w', encoding='utf-8')
    test_data_txt = open(test_list_path, 'w', encoding='utf-8')
    classify = 0
    train_data_num = 0
    test_data_num = 0
    json_name = []
    for item in os.listdir(root_path):
        if not os.path.exists(os.path.join(train_data_path, item)):
            os.makedirs(os.path.join(train_data_path, item))
        if not os.path.exists(os.path.join(test_data_path, item)):
            os.makedirs(os.path.join(test_data_path, item))

        json_name.append(item)
        for file in os.listdir(os.path.join(root_path, item)):
            flag = random.random()

            if flag <= train_rate:
                shutil.copy(os.path.join(root_path, item, file),
                            os.path.join(train_data_path, item))
                print("Copying file from " + os.path.join(root_path, item, file) +
                      " to " + os.path.join(train_data_path, item))
                train_data_txt.write(item + '/' + file + ' ' + str(classify) + "\n")
                train_data_num = train_data_num + 1
            else:
                shutil.copy(os.path.join(root_path, item, file),
                            os.path.join(test_data_path, item))
                print("Copying file from " + os.path.join(root_path, item, file) +
                      " to " + os.path.join(test_data_path, item))
                test_data_txt.write(item + '/' + file + ' ' + str(classify) + "\n")
                test_data_num = test_data_num + 1

        classify = classify + 1

    train_data_txt.close()
    test_data_txt.close()
    print("All done! We found " + str(classify) + " classes!")
    print(str(train_data_num) + " train images.")
    print(str(test_data_num) + " valid images.")

    with open('data/label_to_content.json', 'w') as f:
        f.write('{')
        for i in range(len(json_name)):
            f.write('"' + str(i) + '": "' + json_name[i] + '"')
            if i != len(json_name) - 1:
                f.write(', ')
        f.write('}')


if __name__ == '__main__':
    main()
