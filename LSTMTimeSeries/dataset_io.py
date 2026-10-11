"""Shared battery step-feature reader, retaining the original hour units."""
import pandas as pd
if __package__:
    from .paths import training_workbook
else:
    from paths import training_workbook

def ReadInput(m):

    def time_to_seconds(t):
        return t.hour * 1 + t.minute / 60 + t.second/3600

    # def readfile(m):
    #     # we use this function to read 工步数据
    #     data=pd.read_excel(fr'C:\Users\Lenovo\OneDrive\A_CodePython\MachineLearingQC\TimeSeriesPrediction\train\M0'+m+'.xlsx',sheet_name='工步数据')

    #     # 对于工步数据表格中的“工步号”进行处理
    #     x=data['工步号']
    #     x1=(x-x.min())/(x.max()-x.min())
    #     data['工步号']=x1


    #     # 对 “工步状态”进行处理
    #     label_encoder = LabelEncoder()
    #     data['工步状态'] = label_encoder.fit_transform(data['工步状态'])
    #     # 归一化处理
    #     x=data['工步状态']
    #     x1=(x-x.min())/(x.max()-x.min())
    #     data['工步状态']=x1

    #     # 或许我们直接可以将工步号变成一个1*5的矢量，里面记录总的工作时间

    #     return x1
    
    def readdetails(m):
        # we use this function to read 详细数据

        def create_vector(df):
            # 创建一个1*5的矢量，初始值为0
            vector = [0] * 5
            for _, row in df.iterrows():
                # 工步号从1-5，填入到矢量的对应位置
                if 1 <= row['工步号'] <= 5:
                    vector[row['工步号'] - 1] = row['工步时间_h']
            return vector

        # 我们从详细数据中读取每一步的工作数据
        xls = pd.ExcelFile(training_workbook(m))
        sheet_names = xls.sheet_names

        # 筛选出以 "详细数据" 开头的 sheet 名称
        target_sheets = [sheet for sheet in sheet_names if sheet.startswith("详细数据")]

        # 初始化一个空的 DataFrame
        all_data = pd.DataFrame()

        # 循环读取每个符合条件的 sheet，并将数据整合
        for sheet in target_sheets:
            # 读取当前 sheet
            df = pd.read_excel(xls, sheet_name=sheet)
            
            # 将当前 sheet 的数据添加到 all_data 中
            all_data = pd.concat([all_data, df], ignore_index=True)

        # 读取流程时间
        changes = all_data['工步号'] != all_data['工步号'].shift(-1)

        selected_rows = all_data[changes].copy()
        selected_rows['工步时间_h'] = selected_rows['工步时间'].apply(time_to_seconds)

        # 进一步处理
        result = pd.DataFrame(columns=['循环号', '矢量'])
        rows = []  # 用于暂存每个循环号的结果
        for cycle, group in selected_rows.groupby('循环号'):
            vector = create_vector(group)
            rows.append({'循环号': cycle, '矢量': vector})

        # 使用 pd.concat() 将所有结果整合到 result DataFrame 中
        result = pd.concat([result, pd.DataFrame(rows)], ignore_index=True)
        
        return result

    data=readdetails(m)

    return data
