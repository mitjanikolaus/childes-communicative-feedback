import pandas as pd
from pymer4.models import Lmer

from utils import PROJECT_ROOT_DIR


def glm_caregiver_behavior_clarification_requests(convs):
    # Rescale utt_is_grammatical around 0:
    convs.utt_is_grammatical.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.utt_is_grammatical.min() == -0.5 and convs.utt_is_grammatical.max() == 0.5
    assert convs.response_is_clarification_request.dtype == bool
    # convs_with_response = convs[convs.has_response == True]
    mod = Lmer('response_is_clarification_request ~ utt_is_grammatical * age + (1 | child_name)', family='binomial',
               data=convs)
    print("=" * 50 + "\nCaregiver responses: Clarification requests\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


def glm_caregiver_behavior_acknowledgements(convs):
    # Rescale utt_is_grammatical around 0:
    convs.utt_is_grammatical.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.utt_is_grammatical.min() == -0.5 and convs.utt_is_grammatical.max() == 0.5
    assert convs.response_is_acknowledgement.dtype == bool
    # convs_with_response = convs[convs.has_response == True]
    mod = Lmer('response_is_acknowledgement ~ utt_is_grammatical * age + (1 | child_name)', family='binomial',
               data=convs)
    print("=" * 50 + "\nCaregiver responses: Acknowledgements\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


def glm_child_behavior_clarification_requests_control(convs):
    # Rescale is_follow_up and response_is_clarification_request around 0:
    convs.is_follow_up.replace({False: -0.5, True: 0.5}, inplace=True)
    convs.response_is_clarification_request.replace({False: -0.5, True: 0.5}, inplace=True)

    # Take into account only convs with response
    # assert convs.has_response.dtype == bool
    # convs = convs[convs.has_response == True]

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.response_is_clarification_request.min() == -0.5 and convs.response_is_clarification_request.max() == 0.5
    assert convs.is_grammatical.dtype == bool

    mod = Lmer('is_grammatical ~ response_is_clarification_request * is_follow_up + (1 | child_name) + (1 | age) + (1 | conversation_id)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests (control case)\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


def glm_child_behavior_clarification_requests(convs):
    # Rescale is_follow_up around 0:
    convs.is_follow_up.replace({False: -0.5, True: 0.5}, inplace=True)

    # Take into account only convs with response and that are CR
    # assert convs.has_response.dtype == bool
    assert convs.response_is_clarification_request.dtype == bool
    # convs = convs[convs.has_response == True]
    convs = convs[convs.response_is_clarification_request == True]

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.is_grammatical.dtype == bool

    mod = Lmer('is_grammatical ~ is_follow_up * age + (1 | child_name) + (1 | conversation_id)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


if __name__ == "__main__":
    conversations = pd.read_csv(PROJECT_ROOT_DIR+"/results/grammaticality/conversations.csv", dtype={"error": object, "labels": object})
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 15)

    conversations_melted = pd.read_csv(PROJECT_ROOT_DIR + "/results/grammaticality/conversations_melted.csv", index_col=0, dtype={"error": object, "labels": object})

    # normalize age
    min_age, max_age, mean_age = conversations.age.min(), conversations.age.max(), conversations.age.mean()
    conversations["age"] = (conversations["age"] - mean_age) / (max_age - min_age) * (1 - 0)
    conversations_melted["age"] = (conversations_melted["age"] - mean_age) / (max_age - min_age) * (1 - 0)

    glm_caregiver_behavior_acknowledgements(conversations.copy())
    glm_caregiver_behavior_clarification_requests(conversations.copy())

    glm_child_behavior_clarification_requests_control(conversations_melted.copy())
    glm_child_behavior_clarification_requests(conversations_melted.copy())


