from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd

from analysis_intelligibility import make_plots, perform_per_transcript_analyses, \
    melt_variable


if __name__ == "__main__":
    conversations = pd.read_csv("results/intelligibility/conversations.csv", converters={"pos": literal_eval, "tokens": literal_eval})

    conversations_melted = melt_variable(conversations, "is_intelligible")

    perform_per_transcript_analyses(conversations)

    make_plots(conversations, conversations_melted)

    plt.show()
