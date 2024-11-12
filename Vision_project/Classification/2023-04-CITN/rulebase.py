
def rule_based_order1(result, ppa_t1):
    # result = array([[0.3, 0.7],[0. , 1. ]], dtype=float32)
    # ppa_t1 = array([0., 1., 2., 0., 0., 0.])
    for i in range(len(result.size(0))):
        # if result[i][0] > result[i][1]:
        if ppa_t1[i] == 1:
            result[i] = [0, 1]
    return result


def rule_based_order2(result, ppa_t1, ppa_t2):
    # result = array([[0.3, 0.7],[0. , 1. ]], dtype=float32)
    # ppa_t1 = array([0., 1., 2., 0., 0., 0.])
    for i in range(len(result.size(0))):
        if ppa_t1[i] == 1 or ppa_t2[i] == 1:
            # if result[i][0] > result[i][1]:
            result[i] = [0, 1]
    return result

def rule_based_order3(result, ppa_t1, ppa_t2, ppa_t3):
    # result = array([[0.3, 0.7],[0. , 1. ]], dtype=float32)
    # ppa_t1 = array([0., 1., 2., 0., 0., 0.])
    for i in range(len(result.size(0))):
        if ppa_t1[i] == 1 or ppa_t2[i] == 1 or ppa_t3[i] == 1:
            # if result[i][0] > result[i][1]:
            result[i] = [0, 1]
    return result
    
