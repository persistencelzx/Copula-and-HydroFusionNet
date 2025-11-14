import os
import pandas as pd
import numpy as np
import h5py
from collections import defaultdict

class WatershedDataProcessor:
    def __init__(self, data_folder):
        self.script_dir = os.getcwd()
        self.data_folder = data_folder
        self.model_data_dir = os.path.join(self.script_dir, "model_data")
        os.makedirs(self.model_data_dir, exist_ok=True)

        # 定义流域配置
        self.watersheds_config = {
            '渭河上游': ['临洮', '岷县', '华家岭', '天水', '西吉', '宝鸡'],
            '渭河中游': ['武功', '佛坪', '西安', '镇安'],
            '渭河下游': ['商州', '华山', '铜川'],
            '北洛河': ['洛川', '吴旗', '延安'],
            '泾河': ['环县', '固原', '平凉', '西峰镇', '长武']
        }

        # 定义特征列（输入特征 + 目标变量）
        self.feature_columns = ['year', 'month', 'NDVI', 'SM', 'TMP', 'ET', 'PRE', 'TMN', 'TMX']
        self.target_column = 'target'

        # 列名映射
        self.column_mapping = {
            '参考地名': 'region',
            '年份': 'year',
            '月份': 'month',
            '土壤湿度': 'SM',
            '气温': 'TMP',
            '蒸散量': 'ET',
            '降水量': 'PRE',
            '最低气温': 'TMN',
            '最高气温': 'TMX'
        }

    def evaluate_composite_events(self, df, method='percentile', spi_threshold=None, sti_threshold=None):
        """评估复合事件：使用更灵活的阈值策略"""
        df = df.copy()

        if method == 'percentile':
            # 使用分位数阈值，更适应数据分布
            spi_threshold = spi_threshold or df['SPI'].quantile(0.3)  # 30%分位数
            sti_threshold = sti_threshold or df['STI'].quantile(0.7)  # 70%分位数
            df[self.target_column] = ((df['SPI'] <= spi_threshold) & (df['STI'] >= sti_threshold)).astype(int)
        elif method == 'adaptive':
            # 自适应阈值：基于标准差
            spi_mean, spi_std = df['SPI'].mean(), df['SPI'].std()
            sti_mean, sti_std = df['STI'].mean(), df['STI'].std()
            spi_threshold = spi_mean - 0.5 * spi_std
            sti_threshold = sti_mean + 0.5 * sti_std
            df[self.target_column] = ((df['SPI'] <= spi_threshold) & (df['STI'] >= sti_threshold)).astype(int)
        else:
            # 保持原有方法作为备选
            df[self.target_column] = ((df['SPI'] < 0) & (df['STI'] > 0)).astype(int)

        # 统计信息
        pos_count = df[self.target_column].sum()
        total_count = len(df)
        print(f"    复合事件统计 ({method}): {pos_count}/{total_count} ({pos_count / total_count * 100:.1f}%)")

        return df

    def apply_temporal_aggregation(self, df, window_size=3):
        """应用时间窗口聚合增强正类样本"""
        df = df.copy()
        df = df.sort_values(['region', 'year', 'month']).reset_index(drop=True)

        # 为每个区域单独处理
        balanced_data = []
        for region in df['region'].unique():
            region_df = df[df['region'] == region].copy()

            # 计算滑动窗口内的复合事件密度
            region_df['event_density'] = region_df[self.target_column].rolling(
                window=window_size, min_periods=1, center=True
            ).mean()

            # 如果窗口内事件密度超过阈值，则标记为正类
            region_df[f'{self.target_column}_enhanced'] = (
                    (region_df[self.target_column] == 1) |
                    (region_df['event_density'] >= 0.4)
            ).astype(int)

            balanced_data.append(region_df)

        result_df = pd.concat(balanced_data, ignore_index=True)

        # 更新目标列
        original_pos = result_df[self.target_column].sum()
        enhanced_pos = result_df[f'{self.target_column}_enhanced'].sum()
        result_df[self.target_column] = result_df[f'{self.target_column}_enhanced']
        result_df = result_df.drop(columns=['event_density', f'{self.target_column}_enhanced'])

        print(f"    时间聚合: {original_pos} -> {enhanced_pos} 正类样本")
        return result_df

    def balance_dataset(self, df, balance_ratio=0.15):
        """平衡数据集：确保正类样本达到目标比例"""
        df = df.copy()
        current_ratio = df[self.target_column].mean()

        if current_ratio >= balance_ratio:
            return df

        # 计算需要的正类样本数量
        total_samples = len(df)
        target_positive = int(total_samples * balance_ratio)
        current_positive = df[self.target_column].sum()
        need_positive = target_positive - current_positive

        if need_positive <= 0:
            return df

        # 找到接近正类的样本进行提升
        negative_samples = df[df[self.target_column] == 0].copy()

        # 基于SPI和STI的综合得分选择候选样本
        negative_samples['composite_score'] = (
                                                      -negative_samples['SPI'] + negative_samples['STI']
                                              ) / 2

        # 选择得分最高的样本转为正类
        candidates = negative_samples.nlargest(need_positive, 'composite_score')
        df.loc[candidates.index, self.target_column] = 1

        new_ratio = df[self.target_column].mean()
        print(f"    数据平衡: {current_ratio:.3f} -> {new_ratio:.3f} 正类比例")

        return df

    def load_watershed_data(self, region_names):
        """加载指定流域的所有数据"""
        region_set = set(region_names)
        watershed_data = []

        for filename in os.listdir(self.data_folder):
            if filename.endswith('.csv'):
                region_name = os.path.splitext(filename)[0]
                if region_name in region_set:
                    file_path = os.path.join(self.data_folder, filename)
                    data = pd.read_csv(file_path)
                    watershed_data.append(data)

        return pd.concat(watershed_data, ignore_index=True) if watershed_data else pd.DataFrame()

    def preprocess_watershed_data(self, df, enhance_method='percentile'):
        """预处理单个流域数据"""
        if df.empty:
            return df

        # 重命名列
        df = df.rename(columns=self.column_mapping)

        # 评估复合事件 - 使用改进的方法
        df = self.evaluate_composite_events(df, method=enhance_method)

        # 应用时间聚合增强
        df = self.apply_temporal_aggregation(df)

        # 数据平衡
        df = self.balance_dataset(df, balance_ratio=0.12)

        # 选择需要的列
        required_columns = ['region'] + self.feature_columns + [self.target_column]
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]

        # 按时间排序
        df = df.sort_values(['year', 'month']).reset_index(drop=True)

        return df

    def create_time_aligned_data(self, watershed_data_dict):
        """创建时间对齐的数据集"""
        # 获取所有流域的共同时间范围
        all_time_points = set()
        for df in watershed_data_dict.values():
            if not df.empty:
                time_points = set(zip(df['year'], df['month']))
                all_time_points.update(time_points)

        # 创建完整的时间序列
        sorted_time_points = sorted(all_time_points)
        common_time_df = pd.DataFrame(sorted_time_points, columns=['year', 'month'])

        # 对每个流域数据进行时间对齐
        aligned_data = {}
        for watershed_name, df in watershed_data_dict.items():
            if df.empty:
                continue

            # 按区域分组处理
            region_groups = df.groupby('region')
            aligned_regions = []

            for region_name, region_df in region_groups:
                # 与完整时间序列对齐
                aligned_region = pd.merge(common_time_df, region_df, on=['year', 'month'], how='left')
                aligned_region['region'] = region_name

                # 填充缺失值
                numeric_columns = [col for col in self.feature_columns if
                                   col in aligned_region.columns and col not in ['year', 'month']]
                for col in numeric_columns:
                    aligned_region[col] = aligned_region[col].fillna(aligned_region[col].mean())

                aligned_region[self.target_column] = aligned_region[self.target_column].fillna(0).astype(int)
                aligned_regions.append(aligned_region)

            if aligned_regions:
                aligned_data[watershed_name] = pd.concat(aligned_regions, ignore_index=True)

        return aligned_data

    def prepare_model_data(self, df):
        """准备用于模型训练的数据格式"""
        if df.empty:
            return None, None, None, None

        # 提取特征和目标变量
        feature_data = df[self.feature_columns].values.astype(np.float32)
        target_data = df[self.target_column].values.astype(np.int32)

        # 提取时间索引
        years = df['year'].values.astype(np.int32)
        months = df['month'].values.astype(np.int32)

        return feature_data, target_data, years, months

    def save_processed_data(self, processed_data):
        """保存处理后的数据到HDF5文件"""
        output_file = os.path.join(self.model_data_dir, 'watershed_model_data .h5')

        with h5py.File(output_file, 'w') as hf:
            for watershed_name, df in processed_data.items():
                features, targets, years, months = self.prepare_model_data(df)

                if features is not None:
                    # 创建流域组
                    watershed_group = hf.create_group(watershed_name)

                    # 保存模型数据
                    watershed_group.create_dataset('features', data=features)
                    watershed_group.create_dataset('targets', data=targets)
                    watershed_group.create_dataset('years', data=years)
                    watershed_group.create_dataset('months', data=months)

                    # 保存区域信息
                    regions = df['region'].astype(str).values
                    watershed_group.create_dataset('regions', data=regions,
                                                   dtype=h5py.string_dtype(encoding='utf-8'))

                    # 保存元数据
                    watershed_group.attrs['num_samples'] = len(features)
                    watershed_group.attrs['num_features'] = features.shape[1]
                    watershed_group.attrs['feature_names'] = self.feature_columns
                    watershed_group.attrs['positive_samples'] = int(targets.sum())
                    watershed_group.attrs['negative_samples'] = int(len(targets) - targets.sum())

        print(f"Processed data saved to: {output_file}")
        return output_file

    def process_all_watersheds(self):
        """处理所有流域数据的主要方法"""
        print("Starting watershed data processing...")

        # 加载和预处理各流域数据
        watershed_data = {}
        for watershed_name, region_names in self.watersheds_config.items():
            print(f"Processing {watershed_name}...")
            raw_data = self.load_watershed_data(region_names)
            processed_data = self.preprocess_watershed_data(raw_data, enhance_method='percentile')

            if not processed_data.empty:
                watershed_data[watershed_name] = processed_data
                print(f"  - {len(processed_data)} records processed")
            else:
                print(f"  - No data found for {watershed_name}")

        # 时间对齐
        print("Aligning temporal data across watersheds...")
        aligned_data = self.create_time_aligned_data(watershed_data)

        # 保存处理后的数据
        print("Saving processed data...")
        output_file = self.save_processed_data(aligned_data)

        # 输出摘要信息
        total_samples = sum(len(df) for df in aligned_data.values())
        total_positive = sum(df[self.target_column].sum() for df in aligned_data.values())
        print(f"\nProcessing completed:")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Positive samples: {total_positive}")
        print(f"  - Negative samples: {total_samples - total_positive}")
        print(f"  - Output file: {output_file}")

        return output_file

def main():
    data_folder = os.path.join(os.getcwd(), "final_data")
    processor = WatershedDataProcessor(data_folder)
    processor.process_all_watersheds()

if __name__ == "__main__":
    main()