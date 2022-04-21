import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_linefigures(raw_values, plot_name):
    mean_fitness = []
    std_fitness = []
    mean_fitness_max = []
    std_fitness_max = []
    generations = []

    for gen in pd.unique(raw_values['Generation']):
        generations.append(gen)
        gen_mean = []
        gen_max = []
        gen_values = raw_values.loc[raw_values['Generation'] == gen]
        for run in pd.unique(raw_values['Run']):
            gen_mean.append(gen_values.loc[gen_values['Run'] == run]['Fitness'].mean())
            gen_max.append(gen_values.loc[gen_values['Run'] == run]['Fitness'].max())
        mean_fitness.append(np.mean(gen_mean))
        std_fitness.append(np.std(gen_mean))
        mean_fitness_max.append(np.mean(gen_max))
        std_fitness_max.append(np.std(gen_max))

    plt.plot(generations, mean_fitness, 'b', label="mean fitness")
    plt.errorbar(generations, np.array(mean_fitness), yerr=np.array(std_fitness), fmt='bo', capsize=4)
    #plt.fill_between(generations, np.array(mean_fitness) + np.array(std_fitness), np.array(mean_fitness) - np.array(std_fitness), edgecolor='b', facecolor='b', alpha=0.7)
    plt.plot(generations, mean_fitness_max,'r' , label="avg max fitness")
    #plt.fill_between(generations, np.array(mean_fitness_max) + np.array(std_fitness_max), np.array(mean_fitness_max) - np.array(std_fitness_max), edgecolor='r', facecolor='r', alpha=0.7)
    plt.errorbar(generations, np.array(mean_fitness_max), yerr=np.array(std_fitness_max), fmt='ro', capsize=4)

    plt.legend(['mean fitness', 'mean max fitness'], loc='lower right')
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.suptitle('Generational Fitness')
    plt.xticks(np.arange(0, len(generations), 1))

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig(f'plots/{plot_name}.png')
    plt.show()


def plot_boxplots(gain_df, plot_name):
    # Here, I'm going to do a lot of extra work to specifically calculate the means of the mean of all runs. his is because
    # It's specifically stated in the assignment, and perhaps this leads to different boxplots compared to cal. it directly

    gain = []

    for run in pd.unique(gain_df['Run']):
        gain.append(gain_df.loc[gain_df['Run'] == run]['Gain'].mean())

    fig1, ax1 = plt.subplots()
    ax1.set_title('Gain per EA')
    ax1.boxplot(gain)
    plt.savefig(f'plots/{plot_name}.png')
    plt.show()


class Logger:
    run_list = []
    generations_list = []
    solutions_list = []
    fitness_list = []
    gain_list = []
    enemy_list = []


    @staticmethod
    def get_dataFrame():
        return pd.DataFrame(
            {'Run': Logger.run_list, 'Generation': Logger.generations_list, 'Solution': Logger.solutions_list,
             'Fitness': Logger.fitness_list, 'Gain': Logger.gain_list, "Enemy": Logger.enemy_list})

    @staticmethod
    def reset():
        Logger.run_list = []
        Logger.generations_list = []
        Logger.solutions_list = []
        Logger.fitness_list = []
        Logger.gain_list = []
        Logger.enemy_list = []

    @staticmethod
    def print_lengths():
        print("Run", len(Logger.run_list),
              "Generation", len(Logger.generations_list),
              "Solution", len(Logger.solutions_list),
              "Fitness", len(Logger.fitness_list),
              "Gain", len(Logger.gain_list),
              "Enemy", len(Logger.enemy_list))
