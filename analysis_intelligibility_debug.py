
import matplotlib.pyplot as plt
import pandas as pd

from analysis_intelligibility import make_plots, perform_per_transcript_analyses, \
    melt_variable


if __name__ == "__main__":
    conversations_raw = pd.read_csv("results/conversations_raw.csv")

    conversations = pd.read_csv("results/intelligibility/conversations.csv")

    conversations_melted = melt_variable(conversations, "is_intelligible")

    # normalize age
    min_age, max_age, mean_age = conversations.age.min(), conversations.age.max(), conversations.age.mean()
    conversations["age"] = (conversations["age"] - mean_age) / (max_age - min_age) * (1 - 0)
    conversations_melted["age"] = (conversations_melted["age"] - mean_age) / (max_age - min_age) * (1 - 0)

    perform_per_transcript_analyses(conversations)

    make_plots(conversations, conversations_melted)

    plt.show()
