import React, { useState, useEffect, useContext } from "react";
import {
  View,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Dimensions,
  ImageBackground,
} from "react-native";
import { NavigationProp } from "@react-navigation/native";
import { UserContext } from "../../UserContext";
import { Text } from "galio-framework";
import { Images } from "../../constants";
import { useFocusEffect } from "@react-navigation/native";

interface AnomalyEvent {
  anomaly_create_time: string;
  cctv_id: number;
  anomaly_save_path: string;
  anomaly_delete_yn: boolean;
  log_id: number;
  anomaly_score: number;
  anomaly_feedback: boolean;
  member_id: number;
  cctv_name: string;
  cctv_url: string;
}

const { width, height } = Dimensions.get("screen");

type Tab1ParamList = {
  Tab1Screen: undefined;
  LogDetailScreen: {
    anomaly_create_time: string;
    cctv_id: number;
    anomaly_save_path: string;
    anomaly_delete_yn: boolean;
    log_id: number;
    anomaly_score: number;
    anomaly_feedback: boolean;
    member_id: number;
    cctv_name: string;
    cctv_url: string;
  };
};

interface Tab1ScreenProps {
  navigation: NavigationProp<Tab1ParamList, "Tab1Screen">;
}

function formatDateTime(dateTimeString: string): string {
  const date = new Date(dateTimeString);
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, "0");
  const day = date.getDate().toString().padStart(2, "0");
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");

  return `${year}.${month}.${day} ${hours}:${minutes}:${seconds}`;
}

export default function Tab1Screen(props: Tab1ScreenProps) {
  const { user } = useContext(UserContext);
  // console.log(user)
  const { navigation } = props;
  const [anomalyEvents, setAnomalyEvents] = useState<AnomalyEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<AnomalyEvent[]>([]);
  const [searchText, setSearchText] = useState("");
  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchResults, setSearchResults] = useState<AnomalyEvent[]>([]);
  const itemsPerPage = 4;
  const [performSearch, setPerformSearch] = useState(false);

  const onSearch = () => {
    setPerformSearch((prev) => !prev); // 검색 수행 트리거
  };

  const fetchAnomalyEvents = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.142:30016/api/v0/cctv/loglist_lookup?member_id=${user}`,
        {
          method: "GET",
          headers: { accept: "application/json" },
        },
      );
      console.log("receving data...");
      const data = await response.json();
      console.log(response.ok);

      if (response.ok) {
        console.log(data.isSuccess);
        // console.log(data.result);
        setAnomalyEvents(data.result);
        setTotalPages(Math.ceil(data.result.length / itemsPerPage));
      } else {
        console.error("API 호출에 실패했습니다:", data);
      }
    } catch (error) {
      console.error("API 호출 중 예외가 발생했습니다:", error);
    }
  };
  useFocusEffect(
    React.useCallback(() => {
      fetchAnomalyEvents();
    }, [user]),
  );
  useEffect(() => {
    setCurrentPage(1);
    setTotalPages(Math.ceil(searchResults.length / itemsPerPage));
    setSearchText("");
  }, [searchResults]);

  // 검색 로직
  useEffect(() => {
    const filtered = anomalyEvents.filter((event) => {
      const formattedTime = formatDateTime(
        event.anomaly_create_time,
      ).toLowerCase();
      return (
        event.cctv_name.toLowerCase().includes(searchText.toLowerCase()) ||
        formattedTime.includes(searchText.toLowerCase())
      );
    });

    setSearchResults(filtered); // 검색된 결과를 저장
    setCurrentPage(1); // 페이지를 첫 페이지로 설정
  }, [anomalyEvents, performSearch]);

  // 페이지 변경 로직
  useEffect(() => {
    const endIndex = currentPage * itemsPerPage;
    const startIndex = endIndex - itemsPerPage;
    setFilteredEvents(searchResults.slice(startIndex, endIndex));
  }, [currentPage, searchResults, itemsPerPage]);

  const renderItem = ({ item }: { item: AnomalyEvent }) => (
    <TouchableOpacity
      style={styles.item}
      onPress={() =>
        navigation.navigate("LogDetailScreen", {
          anomaly_create_time: formatDateTime(item.anomaly_create_time),
          cctv_id: item.cctv_id,
          anomaly_save_path: item.anomaly_save_path,
          anomaly_delete_yn: item.anomaly_delete_yn,
          log_id: item.log_id,
          anomaly_score: item.anomaly_score,
          anomaly_feedback: item.anomaly_feedback,
          member_id: item.member_id,
          cctv_name: item.cctv_name,
          cctv_url: item.cctv_url,
        })
      }
    >
      <Text style={{ fontSize: 24, fontFamily: "C24", marginBottom: 5 }}>
        {item.cctv_name}
      </Text>
      <Text style={styles.timestamp}>
        {formatDateTime(item.anomaly_create_time)}
      </Text>
    </TouchableOpacity>
  );

  function controlPage() {
    return (
      <View style={styles.bottomControl}>
        <View style={styles.pageControl}>
          {totalPages > 1 && (
            <>
              {currentPage > 1 ? (
                <TouchableOpacity
                  onPress={() => setCurrentPage(currentPage - 1)}
                >
                  <Text style={styles.pageItem}>‹</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity onPress={() => setCurrentPage(totalPages)}>
                  <Text style={styles.pageItem}>‹</Text>
                </TouchableOpacity>
              )}
              <Text
                style={{
                  margin: 8,
                  padding: 8,
                  minWidth: 50,
                  textAlign: "center",
                }}
              >{`${currentPage}/${totalPages}`}</Text>
              {currentPage < totalPages ? (
                <TouchableOpacity
                  onPress={() => setCurrentPage(currentPage + 1)}
                >
                  <Text style={styles.pageItem}>›</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity onPress={() => setCurrentPage(1)}>
                  <Text style={styles.pageItem}>›</Text>
                </TouchableOpacity>
              )}
            </>
          )}
        </View>
        <View style={{ flex: 1, alignItems: "flex-end" }}>
          <TouchableOpacity
            style={styles.refreshButton}
            onPress={() => fetchAnomalyEvents()}
          >
            <Text style={styles.refreshButtonText}>🔃</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <ImageBackground
      source={Images.Onboarding}
      style={{ width, height, zIndex: 1 }}
    >
      <View style={{ flex: 1 }}>
        <View style={styles.searchContainer}>
          <TextInput
            style={{
              ...styles.searchInput,
              backgroundColor: "white",
              margin: 15,
            }}
            onChangeText={setSearchText}
            value={searchText}
            placeholder="검색 (CCTV 이름 또는 날짜)"
          />
          <TouchableOpacity onPress={onSearch} style={styles.searchButton}>
            <Text>검색</Text>
          </TouchableOpacity>
        </View>
        <FlatList
          data={filteredEvents}
          renderItem={renderItem}
          keyExtractor={(item) => item.log_id.toString()}
          style={{ flex: 1 }}
          contentContainerStyle={{ paddingBottom: 100 }}
          scrollEnabled={false}
        />
        <View style={{ flex: 0.6 }}>{controlPage()}</View>
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  item: {
    backgroundColor: "#FFFFFF", // 회색 배경
    borderWidth: 1,
    borderColor: "#CCCCCC", // 테두리 색상
    borderRadius: 10, // 모서리 둥글게
    padding: 20, // 내부 패딩
    marginVertical: 8,
    marginHorizontal: 16,
    alignItems: "flex-start", // 자식 요소들 왼쪽 정렬
  },
  title: {
    fontSize: 24, // 제목 폰트 사이즈
    fontWeight: "bold", // 글씨 두껍게
    marginBottom: 4, // 제목과 날짜/시간 사이의 여백
    fontFamily: "C24",
  },
  timestamp: {
    fontSize: 16, // 날짜/시간 폰트 사이즈
    color: "#555555", // 날짜/시간 색상
    fontFamily: "NGB",
  },
  bottomControl: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    padding: 10,
    position: "absolute",
    bottom: 180,
    left: 0,
    right: 0,
    backgroundColor: "transparent",
  },
  pageControl: {
    position: "absolute",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
  },
  pageItem: {
    margin: 8,
    padding: 8,
    borderWidth: 0,
    borderColor: "black",
  },
  pageItemActive: {
    backgroundColor: "red",
  },
  searchContainer: {
    flexDirection: "row", // 자식 요소들을 수평으로 나란히 배치
    alignItems: "center", // 자식 요소들을 세로 방향으로 가운데 정렬
    margin: 15,
  },
  searchInput: {
    height: 40,
    borderWidth: 1,
    paddingLeft: 8,
    flex: 1, // 남은 공간을 모두 차지하도록 함
    borderRadius: 10,
    borderColor: "#CCCCCC",
    marginRight: 8, // 검색 버튼과의 간격을 주기 위함
  },
  searchButton: {
    padding: 10,
    backgroundColor: "#ddd", // 버튼의 배경색, 필요에 따라 조정
    borderRadius: 10, // 버튼의 모서리를 둥글게
  },
  refreshButton: {
    marginLeft: 10, // 페이지네이션 버튼과의 간격
    padding: 10,
    borderRadius: 10, // 버튼 모서리 둥글게
  },
  refreshButtonText: {
    color: "#000", // 텍스트 색상
    fontSize: 20,
  },
});
