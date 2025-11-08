#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-MMD Alignment 데이터 생성기
시계열 데이터와 텍스트 데이터를 페어링하여 diffusion 모델 학습용 aligned pairs 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeTextAligner:
    """시계열-텍스트 정렬 및 페어링 클래스"""
    
    def __init__(self, base_path="/home/user/sheoyon/Time-MMD"):
        self.base_path = Path(base_path)
        self.numerical_path = self.base_path / "numerical"
        self.textual_path = self.base_path / "textual"
        self.output_path = self.base_path / "aligned_pairs"
        self.output_path.mkdir(exist_ok=True)
        
        self.categories = [
            'Agriculture', 'Climate', 'Economy', 'Energy', 'Environment',
            'Health_AFR', 'Health_US', 'Security', 'SocialGood', 'Traffic'
        ]
        
    def load_numerical_data(self, category):
        """카테고리별 시계열 데이터 로드"""
        csv_path = self.numerical_path / category / f"{category}.csv"
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path)
        
        # 날짜 컬럼 파싱
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        
        # OT 컬럼 (시계열 값) 정리
        if 'OT' in df.columns:
            df = df[['date', 'start_date', 'end_date', 'OT']].dropna(subset=['date', 'OT'])
            df = df[np.isfinite(df['OT'])]  # 무한대 값 제거
        
        return df
    
    def load_textual_data(self, category, data_type='report'):
        """카테고리별 텍스트 데이터 로드 (report 또는 search)"""
        csv_path = self.textual_path / category / f"{category}_{data_type}.csv"
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path)
        
        # 날짜 컬럼 파싱
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        
        # 유효한 텍스트 데이터만 유지
        if 'fact' in df.columns:
            df = df[df['fact'].notna() & (df['fact'] != 'NA') & (df['fact'] != '')]
        
        return df
    
    def create_time_windows(self, ts_data, lookback_window=96, stride=1):
        """
        시계열 데이터를 일정 개수의 포인트로 윈도우 분할 (sliding window)
        각 포인트의 start_date/end_date를 사용하여 정확한 기간 매칭
        
        Args:
            ts_data: 시계열 DataFrame (date, start_date, end_date, OT 컬럼 포함)
            lookback_window: 윈도우 크기 (데이터 포인트 개수)
            stride: 슬라이딩 스트라이드 (기본값 1 = 한 칸씩 이동)
        
        Returns:
            List of windows with sufficient time series points
        """
        windows = []
        
        if ts_data is None or len(ts_data) == 0:
            return windows
        
        ts_data = ts_data.sort_values('date').reset_index(drop=True)
        
        # 데이터가 lookback_window보다 작으면 전체를 하나의 윈도우로
        if len(ts_data) < lookback_window:
            print(f"    경고: 데이터 길이({len(ts_data)})가 윈도우 크기({lookback_window})보다 작음")
            if len(ts_data) >= 10:  # 최소 10개는 있어야 의미 있음
                # 첫 포인트의 start_date부터 마지막 포인트의 end_date까지
                window_start = ts_data['start_date'].iloc[0] if 'start_date' in ts_data.columns else ts_data['date'].iloc[0]
                window_end = ts_data['end_date'].iloc[-1] if 'end_date' in ts_data.columns else ts_data['date'].iloc[-1]
                
                windows.append({
                    'start_date': window_start,
                    'end_date': window_end,
                    'dates': ts_data['date'].tolist(),
                    'values': ts_data['OT'].tolist(),
                    'length': len(ts_data)
                })
            return windows
        
        # Sliding window로 연속된 lookback_window 개수만큼 추출
        for i in range(0, len(ts_data) - lookback_window + 1, stride):
            window_data = ts_data.iloc[i:i + lookback_window]
            
            # 첫 번째 포인트의 start_date부터 마지막 포인트의 end_date까지
            window_start = window_data['start_date'].iloc[0] if 'start_date' in window_data.columns else window_data['date'].iloc[0]
            window_end = window_data['end_date'].iloc[-1] if 'end_date' in window_data.columns else window_data['date'].iloc[-1]
            
            windows.append({
                'start_date': window_start,
                'end_date': window_end,
                'dates': window_data['date'].tolist(),
                'values': window_data['OT'].tolist(),
                'length': len(window_data)
            })
        
        return windows
    
    def find_matching_text(self, time_window, text_data):
        """
        시계열 윈도우에 매칭되는 텍스트 데이터 찾기
        윈도우의 start_date ~ end_date 기간에 포함되는 모든 텍스트 매칭
        
        Args:
            time_window: {'start_date', 'end_date', 'values', ...}
            text_data: 텍스트 DataFrame (start_date 컬럼 포함)
        
        Returns:
            매칭되는 텍스트 리스트
        """
        if text_data is None or len(text_data) == 0:
            return []
        
        matching_texts = []
        
        window_start = time_window['start_date']
        window_end = time_window['end_date']
        
        for idx, row in text_data.iterrows():
            text_date = row.get('start_date')
            
            if pd.isna(text_date):
                continue
            
            # 텍스트의 날짜가 시계열 윈도우 기간 내에 있는지 확인
            # 윈도우의 start_date ~ end_date 사이에 있으면 매칭
            if window_start <= text_date <= window_end:
                matching_texts.append({
                    'date': text_date,
                    'fact': row.get('fact', ''),
                    'preds': row.get('preds', '')
                })
        
        return matching_texts
    
    def create_aligned_pairs(self, category, lookback_window=96, stride=48, data_type='report'):
        """
        카테고리별 시계열-텍스트 aligned pairs 생성
        
        Args:
            category: 데이터 카테고리
            lookback_window: 시계열 윈도우 크기 (데이터 포인트 개수)
            stride: 슬라이딩 윈도우 스트라이드
            data_type: 텍스트 데이터 타입 ('report' 또는 'search')
        
        Returns:
            aligned pairs 리스트
        """
        print(f"\n[{category}] Aligned pairs 생성 중...")
        
        # 데이터 로드
        ts_data = self.load_numerical_data(category)
        text_data = self.load_textual_data(category, data_type)
        
        if ts_data is None:
            print(f"  ✗ 시계열 데이터 없음")
            return []
        
        if text_data is None:
            print(f"  ✗ 텍스트 데이터 없음")
            return []
        
        print(f"  - 시계열 데이터: {len(ts_data)} 포인트")
        print(f"  - 텍스트 데이터: {len(text_data)} 개")
        
        # 시계열 윈도우 생성
        time_windows = self.create_time_windows(ts_data, lookback_window, stride)
        print(f"  - 생성된 윈도우: {len(time_windows)} 개 (lookback={lookback_window}, stride={stride})")
        
        # 각 윈도우에 대해 매칭되는 텍스트 찾기
        aligned_pairs = []
        pairs_with_text = 0
        
        for window in time_windows:
            matching_texts = self.find_matching_text(window, text_data)
            
            # 하나의 윈도우에 매칭되는 모든 텍스트를 하나의 세트로
            if len(matching_texts) > 0:
                pair = {
                    'category': category,
                    'window_start': window['start_date'].strftime('%Y-%m-%d'),
                    'window_end': window['end_date'].strftime('%Y-%m-%d'),
                    'time_series': {
                        'dates': [d.strftime('%Y-%m-%d') for d in window['dates']],
                        'values': window['values'],
                        'length': window['length']
                    },
                    'texts': [
                        {
                            'date': text['date'].strftime('%Y-%m-%d'),
                            'fact': text['fact'],
                            'prediction': text['preds']
                        }
                        for text in matching_texts
                    ],
                    'num_texts': len(matching_texts),
                    'data_type': data_type
                }
                aligned_pairs.append(pair)
                pairs_with_text += 1
        
        print(f"  ✓ 생성된 aligned pairs: {len(aligned_pairs)} 개")
        print(f"  ✓ 텍스트 있는 윈도우: {pairs_with_text}/{len(time_windows)} ({pairs_with_text/len(time_windows)*100:.1f}%)")
        
        return aligned_pairs
    
    def create_all_aligned_pairs(self, lookback_windows=[48, 96], strides=[24, 48], data_types=['report']):
        """
        모든 카테고리에 대해 aligned pairs 생성
        
        Args:
            lookback_windows: 다양한 윈도우 크기 리스트 (데이터 포인트 개수)
            strides: 슬라이딩 윈도우 스트라이드 리스트
            data_types: 텍스트 데이터 타입 리스트
        """
        print("="*70)
        print("Time-MMD Aligned Pairs 생성 시작")
        print("="*70)
        
        all_results = {}
        
        for lookback in lookback_windows:
            for stride in strides:
                for data_type in data_types:
                    key = f"lookback_{lookback}_stride_{stride}_{data_type}"
                    all_results[key] = {}
                    
                    print(f"\n### Lookback: {lookback} 포인트, Stride: {stride}, Data Type: {data_type} ###")
                    
                    for category in self.categories:
                        pairs = self.create_aligned_pairs(category, lookback, stride, data_type)
                        all_results[key][category] = pairs
        
        # 결과 저장
        self.save_results(all_results)
        
        # 통계 출력
        self.print_statistics(all_results)
        
        return all_results
    
    def save_results(self, results):
        """결과를 JSON 파일로 저장"""
        print("\n" + "="*70)
        print("결과 저장 중...")
        
        for key, category_pairs in results.items():
            output_file = self.output_path / f"{key}.json"
            
            # JSON 직렬화 가능하도록 변환
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(category_pairs, f, ensure_ascii=False, indent=2)
            
            total_pairs = sum(len(pairs) for pairs in category_pairs.values())
            print(f"  ✓ {output_file.name}: {total_pairs} pairs 저장")
        
        print(f"\n모든 결과가 {self.output_path}에 저장되었습니다!")
    
    def print_statistics(self, results):
        """통계 정보 출력"""
        print("\n" + "="*70)
        print("통계 요약")
        print("="*70)
        
        for key, category_pairs in results.items():
            print(f"\n[{key}]")
            
            total_pairs = 0
            for category, pairs in category_pairs.items():
                num_pairs = len(pairs)
                total_pairs += num_pairs
                if num_pairs > 0:
                    print(f"  - {category}: {num_pairs} pairs")
            
            print(f"  → 총합: {total_pairs} pairs")
    
    def create_sample_visualization(self, num_samples=3):
        """샘플 페어 시각화 (JSON으로 저장)"""
        print("\n" + "="*70)
        print(f"샘플 {num_samples}개 추출 중...")
        
        samples = {}
        
        for category in self.categories[:3]:  # 처음 3개 카테고리만
            pairs = self.create_aligned_pairs(category, lookback_window=36, stride=12, data_type='report')
            
            if len(pairs) > 0:
                # 랜덤하게 샘플 선택
                sample_indices = np.random.choice(
                    len(pairs), 
                    min(num_samples, len(pairs)), 
                    replace=False
                )
                
                samples[category] = [pairs[i] for i in sample_indices]
        
        # 샘플 저장
        sample_file = self.output_path / "samples.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 샘플이 {sample_file}에 저장되었습니다!")
        
        # 샘플 프리뷰
        print("\n### 샘플 프리뷰 ###")
        for category, category_samples in samples.items():
            print(f"\n[{category}]")
            for i, sample in enumerate(category_samples[:1]):  # 첫 번째 샘플만
                print(f"\n  Sample {i+1}:")
                print(f"    기간: {sample['window_start']} ~ {sample['window_end']}")
                print(f"    시계열 길이: {sample['time_series']['length']} 포인트")
                print(f"    시계열 값 (처음 5개): {sample['time_series']['values'][:5]}")
                print(f"    매칭된 텍스트 개수: {sample['num_texts']} 개")
                if sample['num_texts'] > 0:
                    print(f"    첫 번째 텍스트 날짜: {sample['texts'][0]['date']}")
                    print(f"    첫 번째 텍스트 Fact: {sample['texts'][0]['fact'][:200]}...")
                    if sample['num_texts'] > 1:
                        print(f"    마지막 텍스트 날짜: {sample['texts'][-1]['date']}")
                        print(f"    마지막 텍스트 Fact: {sample['texts'][-1]['fact'][:200]}...")


def main():
    """메인 함수"""
    print("Time-MMD Alignment 데이터 생성기")
    print("Diffusion 기반 Multi-modal Alignment를 위한 데이터 페어링\n")
    
    aligner = TimeTextAligner()
    
    # 적절한 윈도우 크기 선택
    # 카테고리별 데이터 길이가 다르므로 여러 크기 생성
    # 24개월=2년, 36개월=3년, 48개월=4년
    lookback_windows = [24, 36, 48]  # 데이터 포인트 개수
    strides = [12, 24]  # 오버랩을 위한 stride (50% 오버랩)
    data_types = ['report']  # 먼저 report만
    
    results = aligner.create_all_aligned_pairs(
        lookback_windows=lookback_windows,
        strides=strides,
        data_types=data_types
    )
    
    # 샘플 시각화
    aligner.create_sample_visualization(num_samples=5)
    
    print("\n" + "="*70)
    print("✓ 모든 작업이 완료되었습니다!")
    print("="*70)
    print("\n다음 단계:")
    print("1. aligned_pairs/ 폴더에서 생성된 페어 확인")
    print("2. Diffusion 모델 학습용 데이터로더 구현")
    print("3. Time-series encoder와 Text encoder 설계")
    print("4. Cross-modal alignment loss 정의")
    print("\n생성된 윈도우:")
    print("- Lookback 24: 24개 연속 포인트 (월별이면 2년)")
    print("- Lookback 36: 36개 연속 포인트 (월별이면 3년)")
    print("- Lookback 48: 48개 연속 포인트 (월별이면 4년)")
    print("- Stride 12/24: 50%~100% 오버랩으로 다양한 샘플 생성")


if __name__ == "__main__":
    main()

