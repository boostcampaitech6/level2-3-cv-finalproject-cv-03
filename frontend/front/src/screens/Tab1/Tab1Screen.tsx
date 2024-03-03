import React, { useEffect, useState } from 'react'
import { View, Text, FlatList, StyleSheet, TouchableOpacity } from 'react-native';
import { NavigationProp } from '@react-navigation/native';

interface AnomalyEvent {
  log_id: number;
  anomaly_create_time: string;
  anomaly_save_path: string;
  cctv_id: number;
  cctv_name: string;
  cctv_url: string;
}

interface Tab1ScreenProps {
  navigation: NavigationProp<any>;
}

function formatDateTime(dateTimeString: string): string {
  const date = new Date(dateTimeString);
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');

  return `${year}.${month}.${day} ${hours}:${minutes}:${seconds}`;
}

const member_id = 13

export default function Tab1Screen(props: Tab1ScreenProps) {
  const { navigation } = props;
  const [anomalyEvents, setAnomalyEvents] = useState<AnomalyEvent[]>([]);

  useEffect(() => {
    const fetchAnomalyEvents = async () => {
      try {
        const response = await fetch(`http://10.28.224.142:30016/api/v0/cctv/loglist_lookup?member_id=${member_id}`, {
          method: "GET",
          headers: { 'accept': 'application/json' },
          });
        console.log('receving data...');
        const data = await response.json();
        console.log(response.ok)
  
        if (response.ok) {
          console.log(data.isSuccess);
          console.log(data.result);
          setAnomalyEvents(data.result);
        } else {
          console.error('API 호출에 실패했습니다:', data);
        }
      } catch (error) {
        console.error('API 호출 중 예외가 발생했습니다:', error);
      }
    };
  
    fetchAnomalyEvents();
  }, []);

  const renderItem = ({ item }: { item: AnomalyEvent }) => (
    <TouchableOpacity 
      style={styles.item} 
      onPress={() => navigation.navigate('LogDetailScreen', 
      { log_id: item.log_id, 
        anomaly_create_time: formatDateTime(item.anomaly_create_time),
        anomaly_save_path: item.anomaly_save_path,
        cctv_id: item.cctv_id,
        cctv_name: item.cctv_name,
        cctv_url: item.cctv_url
         })}
    >
      <Text style={styles.title}>{item.cctv_name}</Text>
      <Text style={styles.timestamp}>{formatDateTime(item.anomaly_create_time)}</Text>
    </TouchableOpacity>
  );

  return (
    <FlatList
      data={anomalyEvents}
      renderItem={renderItem}
      keyExtractor={item => item.log_id.toString()}
      style={{ flex: 1 }}
    />
  );
};

const styles = StyleSheet.create({
  // 기존의 styles 정의...
  item: {
    backgroundColor: '#EEEEEE', // 회색 배경
    borderWidth: 1,
    borderColor: '#CCCCCC', // 테두리 색상
    borderRadius: 10, // 모서리 둥글게
    padding: 20, // 내부 패딩
    marginVertical: 8,
    marginHorizontal: 16,
    alignItems: 'flex-start', // 자식 요소들 왼쪽 정렬
  },
  title: {
    fontSize: 24, // 제목 폰트 사이즈
    fontWeight: 'bold', // 글씨 두껍게
    marginBottom: 4, // 제목과 날짜/시간 사이의 여백
  },
  timestamp: {
    fontSize: 16, // 날짜/시간 폰트 사이즈
    color: '#555555', // 날짜/시간 색상
  },
});