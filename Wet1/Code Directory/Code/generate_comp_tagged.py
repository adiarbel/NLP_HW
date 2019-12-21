
import train_helper
import pickle
import test_helper
import feature_statistics_class
import feature2id_class
import numpy as np
import multiprocessing
import argparse




if __name__ == '__main__':



    train1_path = r"../Data/train1.wtag"
    test1_path = r"../Data/test1.wtag"
    train2_path = r"../Data/train2.wtag"
    #test2_path = r"..\Data\test2.wtag"
    comp1_path = r"../Data/comp1.words"
    comp2_path = r"../Data/comp2.words"
    model = -1
    comp_flag = -1
    train_flag = -1
    run_flag = True

    while(run_flag):
        input_flags = True
        model_input = True
        comp_input = True
        train_input = True
        while model_input or comp_input or train_input:
            while model_input:
                print('Insert 1 for model 1 and 2 for model 2\n')
                model = input()
                try:
                    model = int(model)
                    if model != 1 and model != 2:
                        print("Please insert 1 or 2 for model\n")
                    else:
                        model_input = False
                except ValueError:
                    print("Please insert 1 or 2 for model\n")
            while comp_input:
                print('Insert 0 for competition file generation or 1 for running on test 1\n')
                comp_flag = input()
                try:
                    comp_flag = int(comp_flag)
                    if comp_flag != 0 and comp_flag != 1:
                        print("Please insert 0 or 1\n")
                    else:
                        comp_input = False
                except ValueError:
                    print("Please insert 0 or 1\n")
            while train_input:
                print('Insert 0 for pretrain weight file or 1 for running training again 1\n')
                train_flag = input()
                try:
                    train_flag = int(train_flag)
                    if train_flag != 0 and train_flag != 1:
                        print("Please insert 0 or 1\n")
                    else:
                        train_input = False
                except ValueError:
                    print("Please insert 0 or 1\n")

        train_path = ""
        test_path = ""

        if model == 1:
            train_path = train1_path
        elif model == 2:
            train_path = train2_path
        else:
            exit(1)

        if comp_flag == 0:
            if model == 1:
                test_path = comp1_path
            elif model == 2:
                test_path = comp2_path
        elif comp_flag == 1:
            test_path = test1_path  # note that there are tags in test1 that do not appear in train2
        else:
            exit(1)

        ############################
        # Best until now -
        # lambda=1
        # threshold = 1
        # num_iterations = 100
        # milestone_train1 = 85
        ############################

        threshold = 1
        num_iterations = 100
        milestone = 85

        feature_statistics = feature_statistics_class.feature_statistics_class()
        feature_ids = feature2id_class.feature2id_class(feature_statistics, threshold)

        feature_statistics.get_all_counts(train_path)
        all_tags = list(feature_statistics.unigram_tags_count_dict.keys())
        all_tags.insert(0, '*')
        feature_ids.get_all_ids(train_path)

        if train_flag == 1:
            if model == 1:
                weights_path_load = "weights/train1.pkl"
            elif model == 2:
                weights_path_load = "weights/train2.pkl"
            train_helper.train(train_path, threshold, num_iterations, weights_path_load)
        elif train_flag == 0:
            if model == 1:
                weights_path_load = "weights/pretrain1.pkl"
            elif model == 2:
                weights_path_load = "weights/pretrain2.pkl"
        else:
            exit(1)

        #
        with open(weights_path_load, 'rb') as f:  #
            optimal_params = pickle.load(f)
            pre_trained_weights = optimal_params


        if comp_flag == 0:
            print('Comp is generated for model'+str(model))
            test_helper.memm_viterbi_untagged(all_tags, test_path, pre_trained_weights, feature_ids,model)
            print('Prediction is over!')
        elif comp_flag == 1:
            print('Prediction is starting')
            final_tags, confusion_matrix, final_acc = test_helper.memm_viterbi(all_tags, test_path, pre_trained_weights, feature_ids)
            con_mat = (confusion_matrix, final_tags, final_acc)
            print('Prediction is over! Do you want to see the confusion matrix?')
            answer = " "
            while answer not in ("y", "n"):
                answer = input("Enter y/n: ")
                if answer == "y":
                    print(con_mat)
                elif answer == "n":
                    print('')
                else:
                    print("Please enter yes or no.")
                        # Add output function here

        while answer not in ("y", "n"):
            answer = " "
            answer = input("Do you want to exit y/n: ")
            if answer == "y":
                run_flag = False
            elif answer == "n":
                run_flag = True
            else:
                print("Please enter yes or no.")
                # Add output function here

