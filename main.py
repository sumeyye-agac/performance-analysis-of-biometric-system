import numpy as np
import matplotlib.pyplot as plt

def main():

    # load data files
    data_sm = np.loadtxt("./data/data_SM.txt", delimiter=",")
    data_class_labels = np.loadtxt("./data/data_Class_Labels.txt")

    # initialize some variables for simplicity
    number_of_data_points = len(data_sm)
    number_of_person = len(set(data_class_labels))
    number_of_occurences = int(number_of_data_points/number_of_person)

    # split similarity matrix (SM) into genuine and impostor matrices
    genuine_scores, impostor_scores = [], []
    for i in range(0,number_of_data_points,number_of_occurences):
        before_sub_genuine, after_sub_genuine = [], []

        # split genuine scores
        sub_genuine = data_sm[i:i+number_of_occurences,i:i+number_of_occurences]
        genuine_scores.extend(sub_genuine)

        # split impostor scores
        if i == 0:
            after_sub_genuine = data_sm[i:i + number_of_occurences, i + number_of_occurences:number_of_data_points]
            impostor_scores.extend(after_sub_genuine)
        elif i+number_of_occurences ==  number_of_data_points:
            before_sub_genuine = data_sm[i:i + number_of_occurences, 0:i]
            impostor_scores.extend(before_sub_genuine)
        else:
            before_sub_genuine = data_sm[i:i + number_of_occurences, 0:i]
            after_sub_genuine = data_sm[i:i + number_of_occurences, i + number_of_occurences:number_of_data_points]
            impostor_scores.extend(np.concatenate((before_sub_genuine, after_sub_genuine), axis=1))

    # remove NaN elements (diagonal of SM) which not signify any information (i.e. similarity of X and X)
    genuine_scores = np.asarray(genuine_scores)[~np.isnan(np.asarray(genuine_scores))]

    impostor_scores = np.asarray(impostor_scores)

    # find the highest and lowest scores in SM matrix to obtain the range of threshold
    max_threshold = max(max(genuine_scores), max(map(max, impostor_scores)))
    min_threshold = min(min(genuine_scores), min(map(min, impostor_scores)))

    # starting from min_threshold value, calculate FAR and FRR rates for each threshold value up to max_threshold
    y_FRR, y_FAR, t = [], [], []
    for i in list(np.arange(min_threshold,max_threshold,(max_threshold-min_threshold)/1000)):
        genuine_counter, impostor_counter = 0, 0
        for j in range(len(genuine_scores)):
            genuine_counter += (np.asarray(genuine_scores[j]) > i).sum()
        for j in range(len(impostor_scores)):
            impostor_counter += (np.asarray(impostor_scores[j]) > i).sum()
        t.append(i)
        y_FRR.append(1-(genuine_counter/len(genuine_scores)))
        y_FAR.append(impostor_counter/( len(impostor_scores) * len(impostor_scores[0])))

    EER_threshold = 0
    EER_threshold_after_index = 0
    EER = 0
    for i in range(len(y_FAR)):
        if y_FAR[i] == y_FRR[i]:
            EER_threshold, EER = t[i], y_FAR[i]
            break
        if y_FAR[i] < y_FRR[i]: # we passed the intersection point
            EER_threshold_after_index = i
            break

    if EER == 0:
        EER_threshold = (t[EER_threshold_after_index] + t[EER_threshold_after_index-1])/2
        EER = (y_FAR[EER_threshold_after_index] + y_FRR[EER_threshold_after_index]
               + y_FAR[EER_threshold_after_index-1] + y_FRR[EER_threshold_after_index-1])/4

    def get_indices(points, value):
        k = (np.abs(np.array(points)-value)).argmin()
        if points[k] > value:
            return [k+1, k+2]
        elif points[k] < value:
            return [k, k+1]
        else:
            return [k, k]

    # provide FRR values at the following FAR points: FAR=10%, FAR=1%, FAR=0.1%
    FAR_points = [0.1, 0.01, 0.001]
    corresponding_FRR_values = []
    for x in FAR_points:
        position = get_indices(y_FAR, x)
        corresponding_FRR_values.append((y_FRR[position[0]] + y_FRR[position[1]]) / 2)


    print("-"*40)
    print("EER_threshold: {:.2f}".format(EER_threshold))
    print("EER: {:.2f}".format(EER*100) + "%")
    print("-"*40)
    print("FAR = 10%  ------> FRR value: {:.2f}".format(corresponding_FRR_values[0]*100) + "%")
    print("FAR = 1%   ------> FRR value: {:.2f}".format(corresponding_FRR_values[1]*100) + "%")
    print("FAR = 0.1% ------> FRR value: {:.2f}".format(corresponding_FRR_values[2]*100) + "%")

    # FRR and FAR curves with EER point
    plt.plot(t, y_FRR, label="FRR", color='b')
    plt.plot(t, y_FAR, label="FAR", color='r')
    plt.plot(EER_threshold, EER, 'o', label="EER", ms=4, color="g")
    plt.legend()
    plt.show()

    # Genuine and Impostor Score Distribution
    plt.hist(genuine_scores, bins=t, label="genuine", density=1, color="r", alpha=0.5)
    plt.hist(impostor_scores.reshape(impostor_scores.shape[0]*impostor_scores.shape[1]), bins=t, density=1, label="impostor", alpha=0.5)
    plt.legend()
    plt.show()

    # ROC Curve
    plt.plot(y_FRR, 1-np.array(y_FAR), label="ROC Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
