import pandas as pd
from pymer4.models import Lmer
from pymer4.models import Lm

from utils import PROJECT_ROOT_DIR


def glm_caregiver_behavior_clarification_requests(convs):
    # Rescale utt_is_grammatical around 0:
    convs.utt_is_grammatical.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.utt_is_grammatical.min() == -0.5 and convs.utt_is_grammatical.max() == 0.5
    assert convs.response_is_clarification_request.dtype == bool
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

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.response_is_clarification_request.min() == -0.5 and convs.response_is_clarification_request.max() == 0.5
    assert convs.is_grammatical.dtype == bool

    mod = Lmer('is_grammatical ~ response_is_clarification_request * is_follow_up + (1 | child_name) + (1 | age) + (1 | conversation_id)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests (control case)\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


def glm_child_behavior_clarification_requests_control_by_cr_type(convs):
    # Rescale is_follow_up and response_is_clarification_request around 0:
    convs.is_follow_up.replace({False: -0.5, True: 0.5}, inplace=True)
    convs.response_is_clarification_request.replace({False: -0.5, True: 0.5}, inplace=True)
    convs.response_is_repetition_clarification_request.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.response_is_clarification_request.min() == -0.5 and convs.response_is_clarification_request.max() == 0.5
    assert convs.is_grammatical.dtype == bool

    mod = Lmer('is_grammatical ~ response_is_clarification_request * response_is_repetition_clarification_request * is_follow_up + (1 | child_name) + (1 | age) + (1 | conversation_id)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests (control case)\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


def glm_child_behavior_clarification_requests(convs):
    # Rescale is_follow_up around 0:
    convs.is_follow_up.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.response_is_clarification_request.dtype == bool
    convs = convs[convs.response_is_clarification_request == True]

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.is_grammatical.dtype == bool

    mod = Lmer('is_grammatical ~ is_follow_up * age + (1 | child_name) + (1 | conversation_id)', family='binomial', data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


def glm_child_behavior_clarification_requests_by_cr_type(convs):
    # Rescale is_follow_up around 0:
    convs.is_follow_up.replace({False: -0.5, True: 0.5}, inplace=True)

    assert convs.response_is_clarification_request.dtype == bool
    convs = convs[convs.response_is_clarification_request == True]

    assert convs.is_follow_up.min() == -0.5 and convs.is_follow_up.max() == 0.5
    assert convs.is_grammatical.dtype == bool

    mod = Lmer('is_grammatical ~ is_follow_up * response_is_repetition_clarification_request * age + (1 | child_name) + (1 | conversation_id)', family='binomial',
               data=convs)
    print("=" * 50 + "\nChild behavior: Effect of clarification requests by CR type\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])


def glm_tests_marcus_1993():
    johnny_1 = [{"utt_is_grammatical": True, "response_is_clarification_request": True, "child_name": "johnny"}] * 6000
    johnny_2 = [{"utt_is_grammatical": True, "response_is_clarification_request": False, "child_name": "johnny"}] * 2000
    johnny_3 = [{"utt_is_grammatical": False, "response_is_clarification_request": True, "child_name": "johnny"}] * 1500
    johnny_4 = [{"utt_is_grammatical": False, "response_is_clarification_request": False, "child_name": "johnny"}] * 500

    billy_1 = [{"utt_is_grammatical": True, "response_is_clarification_request": True, "child_name": "billy"}] * 3000
    billy_2 = [{"utt_is_grammatical": True, "response_is_clarification_request": False, "child_name": "billy"}] * 3000
    billy_3 = [{"utt_is_grammatical": False, "response_is_clarification_request": True, "child_name": "billy"}] * 2000
    billy_4 = [{"utt_is_grammatical": False, "response_is_clarification_request": False, "child_name": "billy"}] * 2000

    data_johnny = pd.DataFrame(johnny_1 + johnny_2 + johnny_3 + johnny_4)
    data_billy = pd.DataFrame(billy_1 + billy_2 + billy_3 + billy_4)

    ratio_johnny_gr = len(data_johnny[data_johnny.response_is_clarification_request & data_johnny.utt_is_grammatical]) / len(data_johnny[data_johnny.utt_is_grammatical])
    ratio_johnny_ungr = len(data_johnny[data_johnny.response_is_clarification_request & ~data_johnny.utt_is_grammatical]) / len(data_johnny[~data_johnny.utt_is_grammatical])

    print(f"ratio_johnny_gr: {ratio_johnny_gr}")
    print(f"ratio_johnny_ungr: {ratio_johnny_ungr}")

    ratio_billy_gr = len(data_billy[data_billy.response_is_clarification_request & data_billy.utt_is_grammatical]) / len(data_billy[data_billy.utt_is_grammatical])
    ratio_billy_ungr = len(data_billy[data_billy.response_is_clarification_request & ~data_billy.utt_is_grammatical]) / len(data_billy[~data_billy.utt_is_grammatical])

    print(f"ratio_billy_gr: {ratio_billy_gr}")
    print(f"ratio_billy_ungr: {ratio_billy_ungr}")

    data = pd.DataFrame(johnny_1 + johnny_2 + johnny_3 + johnny_4 + billy_1 + billy_2 + billy_3 + billy_4)

    ratio_gr = len(
        data[data.response_is_clarification_request & data.utt_is_grammatical]) / len(
        data[data.utt_is_grammatical])
    ratio_ungr = len(
        data[data.response_is_clarification_request & ~data.utt_is_grammatical]) / len(
        data[~data.utt_is_grammatical])

    print(f"ratio_gr: {ratio_gr}")
    print(f"ratio_ungr: {ratio_ungr}")

    data.response_is_clarification_request.replace({False: 0, True: 1}, inplace=True)

    mod = Lm('response_is_clarification_request ~ utt_is_grammatical', family='binomial', data=data)

    print("=" * 50 + "\nTEST LM Caregiver responses: Clarification requests\n" + "=" * 50)
    fitted = mod.fit()
    print(fitted)
    print(fitted[["Estimate", "SE", "Sig"]])

    mod = Lmer('response_is_clarification_request ~ utt_is_grammatical + (1 | child_name)', family='binomial',
               data=data)
    print("=" * 50 + "\nTEST LMER Caregiver responses: Clarification requests\n" + "=" * 50)
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

    conversations.drop(columns=["error", "labels"], inplace=True)
    conversations_melted.drop(columns=["labels"], inplace=True)

    glm_caregiver_behavior_clarification_requests(conversations.copy())
    glm_caregiver_behavior_acknowledgements(conversations.copy())

    glm_child_behavior_clarification_requests_control(conversations_melted.copy())
    glm_child_behavior_clarification_requests(conversations_melted.copy())

    glm_child_behavior_clarification_requests_by_cr_type(conversations_melted.copy())
    glm_child_behavior_clarification_requests_control_by_cr_type(conversations_melted.copy())
