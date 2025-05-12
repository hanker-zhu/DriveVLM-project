import re
def calc_metrics_lykong_paligemma(results):

    def extract_meta(input_str):
        global error_predicts
        list_match = re.search(r"\[(.*?)\]", input_str, re.DOTALL)
        if not list_match:
            return []

        content = list_match.group(1)
        tuples_str = re.findall(r"\((.*?)\)", content, re.DOTALL)

        result = []
        for tuple_str in tuples_str:
            parts = tuple_str.split(",", 1)
            if len(parts) != 2:
                result.append((None, None))
            else:
                a, b = parts[0].strip(), parts[1].strip()
                result.append((a, b))
        return result


    def calc_acc():
        total = 0
        correct = 0
        speed_correct = 0
        traj_correct = 0
        for data in results:
            predict = data["predict"]
            gt = data["gt"]
            pred_meta = extract_meta(predict)
            gt_meta = extract_meta(gt)
            total += len(gt_meta)
            for idx in range(min(len(pred_meta), len(gt_meta))):
                pred_speed, pred_traj = pred_meta[idx]
                gt_speed, gt_traj = gt_meta[idx]
                if pred_speed and pred_traj:
                    pred_speed, pred_traj = pred_speed.strip(), pred_traj.strip()
                    gt_speed, gt_traj = gt_speed.strip(), gt_traj.strip()
                    if pred_speed == gt_speed:
                        speed_correct += 1
                    if pred_traj == gt_traj:
                        traj_correct += 1
                    if pred_speed == gt_speed and pred_traj == gt_traj:
                        correct += 1
        print(total, correct, speed_correct, traj_correct)
        return {"acc": correct / total, "speed_acc": speed_correct / total, "traj_acc": traj_correct / total}
    return calc_acc


