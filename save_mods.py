import shutil
import os
import time


def gg_save(logger, global_step, reference_loss):
    '''

    Args:
        logger:
        global_step:
        reference_loss:

    Returns:

    '''
    #本代码的目的，是保存损失率最小的10个模型，10个这个数量，可以在下面 len(files) < 10 这里改。
    #把这个save_mods.py，复制到根目录，然后复制最下面的备注，到train.py 的截图位置，然后取消备注
    #【这个path，是准备保存的目录，目录尾部一定要有/】
    path = 'logs/sovits_lossless/'
    print("现在时间 " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #print("logs/44k/D_{}.pth".format(global_step))
    # 检查文件是否存在
    if os.path.exists("logs/44k/D_{}.pth".format(global_step)) and os.path.exists("logs/44k/G_{}.pth".format(global_step)):

        files = os.listdir(path)
        gg_file_name = str(reference_loss)[7:14] + "_" + str(global_step) + "_"
        #print(path + gg_file_name + "D.pth")
        if len(files) < 2:
            #print('小于10')
            # 如果文件存在，将其复制到目标地址
            shutil.copy("logs/44k/D_{}.pth".format(global_step), path + gg_file_name + "D.pth")
            shutil.copy("logs/44k/G_{}.pth".format(global_step), path + gg_file_name + "G.pth")
            print(f'文件已复制到目标地址！损失率为：{reference_loss}')
        else:
            #print('大于10')
            array_file = []
            for file in files:
                loss, step, type = file.split("_")
                array_file.append({'loss': loss, 'step': step})
            max_loss_member = max(array_file, key=lambda x: x['loss'])

            if float(max_loss_member["loss"]) > reference_loss:
                #print('损失率 新低')
                os.remove(path + max_loss_member['loss'] + "_" + max_loss_member['step'] + "_D.pth")
                os.remove(path + max_loss_member['loss'] + "_" + max_loss_member['step'] + "_G.pth")
                # 保存模型
                shutil.copy("logs/44k/D_{}.pth".format(global_step), path + gg_file_name + "D.pth")
                shutil.copy("logs/44k/G_{}.pth".format(global_step), path + gg_file_name + "G.pth")
                print(f'文件已复制到目标地址！损失率为：{reference_loss}')
            else:
                print("损失率高于已存在的10个，不进行保存。")
    else:
        print('上一步还未保存，文件不存在！')

#【复制下面的，到train.py 的截图位置】

# # 蝈蝈自定义，还有save_mods.py文件
# import save_mods
# import sys
# import traceback
#
# try:
#     save_mods.gg_save(logger, global_step, reference_loss)
# except Exception as e:
#     print(1 / 1)
#     # 这个是输出错误的具体原因
#     print(e)  # 输出：division by zero
#     print(sys.exc_info())  # 输出：(<class 'ZeroDivisionError'>, ZeroDivisionError('division by zero'), <traceback object at 0x000001A1A7B03380>)
#
#     # 以下两步都是输出错误的具体位置，报错行号位置在第几行
#     print('\n', '>>>' * 20)
#     print(traceback.print_exc())
#     print('\n', '>>>' * 20)
#     print(traceback.format_exc())
