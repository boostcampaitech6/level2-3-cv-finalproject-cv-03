import React, { useState, useEffect, useContext } from "react";
import { View, FlatList, StyleSheet, TouchableOpacity, TextInput, Dimensions, ImageBackground } from 'react-native';
import { NavigationProp } from '@react-navigation/native';
import { UserContext } from '../../UserContext';
import { Block, Text, theme } from "galio-framework";
import { Images, argonTheme } from "../../constants";

interface AnomalyEvent {
  anomaly_create_time: string,
  cctv_id: number,
  anomaly_save_path: string,
  anomaly_delete_yn: boolean,
  log_id: number,
  anomaly_score: number,
  anomaly_feedback: boolean,
  member_id: number,
  cctv_name: string,
  cctv_url: string
}

const { width, height } = Dimensions.get("screen");

const thumbMeasure = (width - 48 - 32) / 3;

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

export default function Tab1Screen(props: Tab1ScreenProps) {
  const { user } = useContext(UserContext);
  // console.log(user)
  const { navigation } = props;
  const [anomalyEvents, setAnomalyEvents] = useState<AnomalyEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<AnomalyEvent[]>([]);
  const [searchText, setSearchText] = useState('');

  useEffect(() => {
    const fetchAnomalyEvents = async () => {
      try {
        const response = await fetch(`http://10.28.224.142:30016/api/v0/cctv/loglist_lookup?member_id=${user}`, {
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

  useEffect(() => {
    filterEvents();
  }, [searchText, anomalyEvents]);

  const filterEvents = () => {
    const filtered = anomalyEvents.filter(event => {
      const formattedTime = formatDateTime(event.anomaly_create_time).toLowerCase();
      return event.cctv_name.toLowerCase().includes(searchText.toLowerCase()) || formattedTime.includes(searchText.toLowerCase());
    });
    setFilteredEvents(filtered);
  };

  const renderItem = ({ item }: { item: AnomalyEvent }) => (
    <TouchableOpacity 
      style={styles.item} 
      onPress={() => navigation.navigate('LogDetailScreen', 
      { anomaly_create_time: formatDateTime(item.anomaly_create_time),
        cctv_id: item.cctv_id,
        anomaly_save_path: item.anomaly_save_path,
        anomaly_delete_yn: item.anomaly_delete_yn,
        log_id: item.log_id,
        anomaly_score: item.anomaly_score,
        anomaly_feedback: item.anomaly_feedback,
        member_id: item.member_id,
        cctv_name: item.cctv_name,
        cctv_url: item.cctv_url
         })}
    >
      <Text style={styles.title}>{item.cctv_name}</Text>
      <Text style={styles.timestamp}>{formatDateTime(item.anomaly_create_time)}</Text>
    </TouchableOpacity>
  );

  return (
    <ImageBackground
        source={Images.Onboarding}
        style={{ width, height, zIndex: 1 }}
      >
      <View style={{ flex: 1 }}>
        <TextInput
          style={{...styles.searchInput, backgroundColor: 'white', margin:15}}
          onChangeText={setSearchText}
          value={searchText}
          placeholder="검색 (CCTV 이름 또는 날짜)"
        />
        <FlatList
          data={filteredEvents}
          renderItem={renderItem}
          keyExtractor={item => item.log_id.toString()}
          style={{ flex: 1 }}
        />
      </View>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  searchInput: {
    height: 40,
    borderWidth: 1,
    paddingLeft: 8,
    margin: 10,
    borderRadius: 10,
    borderColor: '#CCCCCC',
  },
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
      fontFamily: 'C24',
  },
  timestamp: {
    fontSize: 16, // 날짜/시간 폰트 사이즈
    color: '#555555', // 날짜/시간 색상
    fontFamily: 'NGB',
  },
});