"""绘制多分类pr曲线
每一种类别都可以得到多个相应的精准率和召回率,多分类问题可以得到多组p, r值
平均类别精准率又称“宏精准率”(macro-P)和“宏召回率”(macro-R)得到的曲线为"宏P-R曲线"
"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import sys
import itertools
import os


def draw_pr_curve(precision_list,
                  recall,
                  iou_list,
                  out_dir='pr_curve',
                  Model = None,
                  file_name='precision_recall_curve.jpg'):
    base_path = os.path.join(out_dir, Model)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    output_path = os.path.join(base_path, file_name)
    print("pr_curve save in ", output_path)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('Matplotlib not found, plaese install matplotlib.'
                     'for example: `pip install matplotlib`.')
        raise e
    plt.cla()
    plt.figure('P-R Curve')
    plt.title('Precision/Recall Curve(IoU={})'.format(iou_list))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    for iou, precision in zip(iou_list, precision_list):
        plt.plot(recall, precision, label="IOU=" + str(iou))
    plt.legend(bbox_to_anchor=(0.5, -0.15),loc=9,ncol=10)
    plt.tight_layout()
    plt.savefig(output_path)


def cocoapi_eval(jsonfile,
                 style,
                 Model,
                 coco_gt=None,
                 anno_file=None,
                 out_dir=None,
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        coco_gt (str): Whether to load COCOAPI through anno_file,eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)do not have 'area', please set use_area=False.
    """
    assert coco_gt != None or anno_file != None
    
    if coco_gt == None:
        coco_gt = COCO(anno_file)
    print("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile) # 加载预测的result.json 按照coco_gt的形式构造coco_dt 可以细看一下这个就相当于实现了我的choose_label_eval.py的一些功能
    
    coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            print(
                'terminaltables not found, plaese install terminaltables. '
                'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']

        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))
            
            # IOU阈值0.5， 0.6， 0.7， 0.8， 0.9
            idx2iou = {0: 0.5, 1:0.55, 2:0.60, 3:0.65, 4:0.70, 5:0.75, 6:0.80, 7:0.85, 8:0.90, 9:0.95}
            pr_array_1 = precisions[0, :, idx, 0, 2]
            pr_array_2 = precisions[2, :, idx, 0, 2]
            pr_array_3 = precisions[4, :, idx, 0, 2]
            pr_array_4 = precisions[6, :, idx, 0, 2]
            pr_array_5 = precisions[8, :, idx, 0, 2]

            pr_array_list = [pr_array_1, pr_array_2, pr_array_3, pr_array_4, pr_array_5]
            recall_array = np.arange(0.0, 1.01, 0.01)
            draw_pr_curve(
                pr_array_list,
                recall_array,
                iou_list=[idx2iou[0], idx2iou[2], idx2iou[4], idx2iou[6], idx2iou[8]],
                out_dir=out_dir,
                Model = Model,
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))
            
        # 所有类别的pr也即“宏P-R曲线” 
        num_categories = len(cat_ids)
        def compute_all_categories_percision(iou_id, num_categories):
            all_pr_array = [0 for i in range(101)]
            exist_num_categories = num_categories
            for idx in range(0, num_categories):
                if precisions[iou_id, :, idx, 0, 2][0] != -1:
                    all_pr_array += precisions[iou_id, :, idx, 0, 2]
                else:
                    exist_num_categories = exist_num_categories - 1
            return all_pr_array / exist_num_categories

        all_pr_array_1 = compute_all_categories_percision(0, num_categories)
        all_pr_array_2 = compute_all_categories_percision(2, num_categories)
        all_pr_array_3 = compute_all_categories_percision(4, num_categories)
        all_pr_array_4 = compute_all_categories_percision(6, num_categories)
        all_pr_array_5 = compute_all_categories_percision(8, num_categories)
        all_pr_array_list = [all_pr_array_1, all_pr_array_2, all_pr_array_3, all_pr_array_4, all_pr_array_5]

        draw_pr_curve(all_pr_array_list,
                      recall_array,
                      iou_list=[idx2iou[0], idx2iou[2], idx2iou[4], idx2iou[6], idx2iou[8]],
                      out_dir=out_dir,
                      Model = Model,
                      file_name='all cagegories precision_recall_curve.jpg')
        
        
        # 绘制各个类别的IOU = 0.5:0.95ap表格
        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            * [results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('Per-category of {} AP: \n{}'.format(style, table.table))
        print("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()

    # 返回评估的各项指标的数值列表
    return coco_eval.stats


def draw_all_model_sample_iou_pr_curve(precision_list,
                                        recall,
                                        iou,
                                        out_dir='pr_curve/all_model',
                                        Model_Name_List = None,
                                        Modify_Model_Name_dict=None,
                                        file_name='precision_recall_curve.jpg'):            
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_path = os.path.join(out_dir, file_name)
    print("pr_curve save in ", output_path)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('Matplotlib not found, plaese install matplotlib.'
                     'for example: `pip install matplotlib`.')
        raise e
    plt.cla()
    plt.figure('P-R Curve')
    plt.title('Precision/Recall Curve(IoU={})'.format(iou))
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.grid(True)
    for precision, Model in zip(precision_list, Model_Name_List):
        plt.plot(recall, precision, label=Modify_Model_Name_dict[Model])
    # plt.legend(bbox_to_anchor=(0.5, -0.15),loc=9,ncol=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def draw_multiple_model_pr_curve(style, Model_Name_List, Modify_Model_Name_dict):

    all_model_pr_array_iou5 = []
    all_model_pr_array_iou6 = []
    all_model_pr_array_iou7 = []
    all_model_pr_array_iou8 = []
    all_model_pr_array_iou9 = []
    all_model_pr_array_iou55 = []
    all_model_pr_array_iou65 = []
    all_model_pr_array_iou75 = []
    all_model_pr_array_iou85 = []
    all_model_pr_array_iou95 = []
    all_model_pr_array_iou5_9 = []
    
    for Model in Model_Name_List:
        anno_file = "eval/" + Model + "/truth.json"
        jsonfile = "eval/" + Model + "/result.json"
        coco_gt = COCO(anno_file)
        coco_dt = coco_gt.loadRes(jsonfile) # 加载预测的result.json 按照coco_gt的形式构造coco_dt 
        
        coco_eval = COCOeval(coco_gt, coco_dt, style)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Compute all-category AP and PR curve
        precisions = coco_eval.eval['precision']

        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]

        
        recall_array = np.arange(0.0, 1.01, 0.01)

            
        # 所有类别的pr也即“宏P-R曲线” 
        num_categories = len(cat_ids)
        def compute_all_categories_percision(iou_id, num_categories):
            all_categories_pr_array = np.zeros(101)
            exist_num_categories = num_categories
            for idx in range(0, num_categories):
                if precisions[iou_id, :, idx, 0, 2][0] != -1:
                    all_categories_pr_array += precisions[iou_id, :, idx, 0, 2]
                else:
                    exist_num_categories = exist_num_categories - 1
            return all_categories_pr_array / exist_num_categories

        single_model_pr_array_iou5 = compute_all_categories_percision(0, num_categories)
        single_model_pr_array_iou55 = compute_all_categories_percision(1, num_categories)
        single_model_pr_array_iou6 = compute_all_categories_percision(2, num_categories)
        single_model_pr_array_iou65 = compute_all_categories_percision(3, num_categories)
        single_model_pr_array_iou7 = compute_all_categories_percision(4, num_categories)
        single_model_pr_array_iou75= compute_all_categories_percision(5, num_categories)
        single_model_pr_array_iou8 = compute_all_categories_percision(6, num_categories)
        single_model_pr_array_iou85 = compute_all_categories_percision(7, num_categories)
        single_model_pr_array_iou9 = compute_all_categories_percision(8, num_categories)
        single_model_pr_array_iou95 = compute_all_categories_percision(9, num_categories)

        all_model_pr_array_iou5.append(single_model_pr_array_iou5)
        all_model_pr_array_iou6.append(single_model_pr_array_iou6)
        all_model_pr_array_iou7.append(single_model_pr_array_iou7)
        all_model_pr_array_iou8.append(single_model_pr_array_iou8)
        all_model_pr_array_iou9.append(single_model_pr_array_iou9)
        all_model_pr_array_iou55.append(single_model_pr_array_iou55)
        all_model_pr_array_iou65.append(single_model_pr_array_iou65)
        all_model_pr_array_iou75.append(single_model_pr_array_iou75)
        all_model_pr_array_iou85.append(single_model_pr_array_iou85)
        all_model_pr_array_iou95.append(single_model_pr_array_iou95)

    all_model_pr_array_iou5_9.append(all_model_pr_array_iou5)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou6)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou7)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou8)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou9)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou55)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou65)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou75)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou85)
    all_model_pr_array_iou5_9.append(all_model_pr_array_iou95)

    idx2iou = {0: 0.5, 1:0.55, 2:0.60, 3:0.65, 4:0.70, 5:0.75, 6:0.80, 7:0.85, 8:0.90, 9:0.95}
    iou_list = [idx2iou[0], idx2iou[1], idx2iou[2], idx2iou[3], idx2iou[4], idx2iou[5], idx2iou[6], idx2iou[7], idx2iou[8], idx2iou[9]]

    for all_model_pr_array_iou, iou in zip(all_model_pr_array_iou5_9, iou_list):
        draw_all_model_sample_iou_pr_curve(all_model_pr_array_iou,
                                            recall_array,
                                            iou=iou,
                                            out_dir=style + '_pr_curve/XJT/all_model',
                                            Model_Name_List=Model_Name_List,
                                            Modify_Model_Name_dict=Modify_Model_Name_dict,
                                            file_name="all model all cagegories iou=" + str(iou) + "precision_recall_curve.jpg")
    
if __name__ == "__main__":
    # 绘制所有的模型同一个iou的pr曲线
    draw_multiple_model_pr_curve('bbox', 
                                 Model_Name_List=["XJT/specific_no_pre",
                                                  "XJT/specific_itk_pre",
                                                  "XJT/specific_buu400_pre",
                                                  "XJT/logic"],
                                Modify_Model_Name_dict={"XJT/specific_no_pre":"Model S1",
                                                        "XJT/specific_itk_pre":"Model S2",
                                                        "XJT/specific_buu400_pre":"Model S3",
                                                        "XJT/logic":"Model L"})
  
    # 绘制单个模型的pr曲线，包括单个类别以及所有类别的iou=0.5，0.6，0.7，0.8，0.9的pr曲线
    cocoapi_eval("eval/XJT/specific_no_pre/result.json", 'bbox', Model="specific_no_pre", coco_gt=None,
                  anno_file="eval/XJT/specific_no_pre/truth.json", out_dir="bbox_pr_curve/XJT", classwise=True)
    
    cocoapi_eval("eval/XJT/specific_itk_pre/result.json", 'bbox', Model="specific_itk_pre", coco_gt=None,
                  anno_file="eval/XJT/specific_itk_pre/truth.json", out_dir="bbox_pr_curve/XJT", classwise=True)
    
    cocoapi_eval("eval/XJT/specific_buu400_pre/result.json", 'bbox', Model="specific_buu400_pre", coco_gt=None,
                  anno_file="eval/XJT/specific_buu400_pre/truth.json", out_dir="bbox_pr_curve/XJT", classwise=True)
    
    cocoapi_eval("eval/XJT/logic/result.json", 'bbox', Model="logic", coco_gt=None,
                anno_file="eval/XJT/logic/truth.json", out_dir="bbox_pr_curve/XJT", classwise=True)
