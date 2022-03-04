import pandas as pd
from pymer4.models import Lmer

from analysis_intelligibility import melt_is_intelligible_variable


def glm_caregiver_behavior_timing(convs):
    # Rescale utt_is_intelligible around 0:
    convs.utt_is_intelligible.replace({False: -0.5, True: 0.5}, inplace=True)
    assert convs.utt_is_intelligible.min() == -0.5 and convs.utt_is_intelligible.max() == 0.5
    assert convs.has_response.dtype == bool
    mod = Lmer('has_response ~ utt_is_intelligible * age + (1 | child_name)', family='binomial', data=convs)
    print("=" * 50 + "\nCaregiver responses: Timing\n" + "=" * 50)
    print(mod.fit())


def glm_caregiver_behavior_clarification_requests(convs):
    # Rescale utt_is_intelligible around 0:
    convs.utt_is_intelligible.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.utt_is_intelligible.min() == -0.5 and convs.utt_is_intelligible.max() == 0.5
    assert convs.response_is_clarification_request.dtype == bool
    convs_with_response = convs[convs.has_response == True]
    mod = Lmer('response_is_clarification_request ~ utt_is_intelligible * age + (1 | child_name)', family='binomial',
               data=convs_with_response)
    print("=" * 50 + "\nCaregiver responses: Clarification requests\n" + "=" * 50)
    print(mod.fit())


def glm_caregiver_behavior_pos_feedback(convs):
    # Rescale utt_is_intelligible around 0:
    convs.utt_is_intelligible.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.utt_is_intelligible.min() == -0.5 and convs.utt_is_intelligible.max() == 0.5
    assert convs.pos_feedback.dtype == bool
    mod = Lmer('pos_feedback ~ utt_is_intelligible * age + (1 | child_name)', family='binomial',
               data=convs)
    print("=" * 50 + "\nCaregiver responses: Positive feedback (= No pause, no clarification request)\n" + "=" * 50)
    print(mod.fit())


def glm_child_behavior_timing(convs):
    # Rescale has_response around 0:
    convs.has_response.replace({False: -0.5, True: 0.5}, inplace=True)

    # Take into account only convs where first child utt is intelligible (positive case)
    assert convs.utt_is_intelligible.dtype == bool
    convs = convs[convs.utt_is_intelligible == True]

    assert convs.has_response.min() == -0.5 and convs.has_response.max() == 0.5
    assert convs.follow_up_is_intelligible.dtype == bool
    mod = Lmer('follow_up_is_intelligible ~ has_response * age + (1 | child_name)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of timing\n" + "=" * 50)
    print(mod.fit())


def glm_child_behavior_clarification_requests_control(convs):
    # Rescale is_follow_up and response_is_clarification_request around 0:
    convs.is_follow_up.replace({False: -0.5, True: 0.5}, inplace=True)
    convs.response_is_clarification_request.replace({False: -0.5, True: 0.5}, inplace=True)

    # Take into account only convs with response
    assert convs.has_response.dtype == bool
    convs = convs[convs.has_response == True]

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.response_is_clarification_request.min() == -0.5 and convs.response_is_clarification_request.max() == 0.5
    assert convs.is_intelligible.dtype == bool

    mod = Lmer('is_intelligible ~ response_is_clarification_request * is_follow_up + (1 | child_name) + (1 | age) + (1 | conversation_id)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests (control case)\n" + "=" * 50)
    print(mod.fit())


def glm_child_behavior_clarification_requests(convs):
    # Rescale is_follow_up around 0:
    convs.is_follow_up.replace({False: -0.5, True: 0.5}, inplace=True)

    # Take into account only convs with response and that are CR
    assert convs.has_response.dtype == bool
    assert convs.response_is_clarification_request.dtype == bool
    convs = convs[convs.has_response == True]
    convs = convs[convs.response_is_clarification_request == True]

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.is_intelligible.dtype == bool

    mod = Lmer('is_intelligible ~ is_follow_up * age + (1 | child_name) + (1 | conversation_id)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests\n" + "=" * 50)
    print(mod.fit())


if __name__ == "__main__":
    conversations = pd.read_csv("results/intelligibility/conversations.csv")

    conversations_melted = melt_is_intelligible_variable(conversations)

    # normalize age
    min_age, max_age, mean_age = conversations.age.min(), conversations.age.max(), conversations.age.mean()
    conversations["age"] = (conversations["age"] - mean_age) / (max_age - min_age) * (1 - 0)
    conversations_melted["age"] = (conversations_melted["age"] - mean_age) / (max_age - min_age) * (1 - 0)

    glm_caregiver_behavior_timing(conversations.copy())
    glm_caregiver_behavior_clarification_requests(conversations.copy())
    # glm_caregiver_behavior_pos_feedback(conversations.copy())

    glm_child_behavior_timing(conversations.copy())
    glm_child_behavior_clarification_requests_control(conversations_melted.copy())
    glm_child_behavior_clarification_requests(conversations_melted.copy())


