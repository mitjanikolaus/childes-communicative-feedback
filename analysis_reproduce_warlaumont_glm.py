import pandas as pd
from pymer4.models import Lmer


def glm_caregiver_behavior_timing(convs):
    # Rescale utt_is_intelligible around 0:
    convs.utt_is_speech_related.replace({False: -0.5, True: 0.5}, inplace=True)
    assert convs.utt_is_speech_related.min() == -0.5 and convs.utt_is_speech_related.max() == 0.5
    assert convs.has_response.dtype == bool
    mod = Lmer('has_response ~ utt_is_speech_related * age + (1 | child_name)', family='binomial', data=convs)
    print("=" * 50 + "\nCaregiver responses: Timing\n" + "=" * 50)
    print(mod.fit())


def glm_child_behavior_timing(convs):
    # Rescale has_response around 0:
    convs.has_response.replace({False: -0.5, True: 0.5}, inplace=True)

    # Take into account only convs where first child utt is speech-related (positive case)
    assert convs.utt_is_speech_related.dtype == bool
    convs = convs[convs.utt_is_speech_related == True]

    assert convs.has_response.min() == -0.5 and convs.has_response.max() == 0.5
    assert convs.follow_up_is_speech_related.dtype == bool
    mod = Lmer('follow_up_is_speech_related ~ has_response * age + (1 | child_name)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of timing\n" + "=" * 50)
    print(mod.fit())


if __name__ == "__main__":
    conversations = pd.read_csv("results/reproduce_warlaumont/conversations.csv")
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 15)

    # normalize age
    min_age, max_age, mean_age = conversations.age.min(), conversations.age.max(), conversations.age.mean()
    conversations["age"] = (conversations["age"] - mean_age) / (max_age - min_age) * (1 - 0)

    glm_caregiver_behavior_timing(conversations.copy())

    glm_child_behavior_timing(conversations.copy())
