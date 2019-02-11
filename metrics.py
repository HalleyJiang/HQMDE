import numpy as np


def compute_metrics_for_multi_maps(gt_depth_maps, pd_depth_maps, cap = None):
    gt_depth_maps = np.squeeze(gt_depth_maps)
    pd_depth_maps = np.squeeze(pd_depth_maps)
    if len(np.shape(gt_depth_maps)) == 2:
        return compute_metrics_for_single_map(gt_depth_maps, pd_depth_maps, cap)
    else:
        num_maps, height, width = np.shape(gt_depth_maps)
        ABS_REL = np.zeros(num_maps, np.float32)
        SQ_REL = np.zeros(num_maps, np.float32)
        RMSE = np.zeros(num_maps, np.float32)
        RMSE_log = np.zeros(num_maps, np.float32)
        Log10 = np.zeros(num_maps, np.float32)
        ACCURACY1 = np.zeros(num_maps, np.float32)
        ACCURACY2 = np.zeros(num_maps, np.float32)
        ACCURACY3 = np.zeros(num_maps, np.float32)
        for i in range(num_maps):
            ABS_REL[i], SQ_REL[i], RMSE[i], RMSE_log[i], Log10[i], ACCURACY1[i], ACCURACY2[i], ACCURACY3[i] = \
                compute_metrics_for_single_map(gt_depth_maps[i], pd_depth_maps[i], cap)
        return ABS_REL.mean(), SQ_REL.mean(), RMSE.mean(), RMSE_log.mean(), Log10.mean(), ACCURACY1.mean(), ACCURACY2.mean(), ACCURACY3.mean()


def compute_metrics_for_single_map(gt_depth_map, pd_depth_map, cap = None):
    # Create mask for valid pixels
    mask = gt_depth_map > 0.01
    gt_depth_map[gt_depth_map <= 0.01] = 0.01 # epsilon
    pd_depth_map[pd_depth_map <= 0.01] = 0.01
    if cap:
        mask_cap = gt_depth_map <= cap
        mask = np.logical_and(mask, mask_cap)

    # Count number of valid pixels
    val_pxls = np.sum(mask)

    # Compute absolute relative error
    abs_rel = np.abs(gt_depth_map - pd_depth_map) / gt_depth_map
    abs_rel[~mask] = 0
    S_abs_rel = np.sum(abs_rel)

    # Compute square relative error
    sq_rel = np.square(gt_depth_map - pd_depth_map) / gt_depth_map
    sq_rel[~mask] = 0
    S_sq_rel = np.sum(sq_rel)

    # Compute root mean square error
    rmse = np.square(gt_depth_map - pd_depth_map)
    rmse[~mask] = 0
    S_rmse = np.sum(rmse)

    # Compute root mean square error log
    rmse_log = np.square(np.log(gt_depth_map) - np.log(pd_depth_map))
    rmse_log[~mask] = 0
    S_rmse_log = np.sum(rmse_log)

    # Compute log10 error
    log10 = np.abs(np.log10(gt_depth_map) - np.log10(pd_depth_map))
    log10[~mask] = 0
    S_log10 = np.sum(log10)

    max_ratio = np.maximum(gt_depth_map / pd_depth_map, pd_depth_map / gt_depth_map)

    # Compute accuracies for different deltas(thresholds)
    acc1 = np.asarray(np.logical_and(max_ratio < 1.25, mask), dtype=np.float32)
    acc2 = np.asarray(np.logical_and(max_ratio < 1.25 ** 2, mask), dtype=np.float32)
    acc3 = np.asarray(np.logical_and(max_ratio < 1.25 ** 3, mask), dtype=np.float32)

    S_acc1 = np.sum(acc1)
    S_acc2 = np.sum(acc2)
    S_acc3 = np.sum(acc3)

    ABS_REL = S_abs_rel / val_pxls
    SQ_REL = S_sq_rel / val_pxls
    RMSE = np.sqrt(S_rmse / val_pxls)
    RMSE_log = np.sqrt(S_rmse_log / val_pxls)
    Log10 = S_log10/val_pxls
    ACCURACY1 = S_acc1 / val_pxls
    ACCURACY2 = S_acc2 / val_pxls
    ACCURACY3 = S_acc3 / val_pxls

    return ABS_REL, SQ_REL, RMSE, RMSE_log, Log10, ACCURACY1, ACCURACY2, ACCURACY3
