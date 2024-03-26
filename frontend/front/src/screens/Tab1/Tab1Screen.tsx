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
import { useFocusEffect } from "@react-navigation/native";
import * as Notifications from "expo-notifications";
import DateTimePicker from "@react-native-community/datetimepicker";
import { Overlay } from "react-native-elements";
import { Images } from "../../constants";

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

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

function formatDateTime(
  dateTimeString: string,
  apicall: boolean = false,
): string {
  const date = new Date(dateTimeString);
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, "0");
  const day = date.getDate().toString().padStart(2, "0");
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");

  if (apicall) {
    return `${year}-${month}-${day}%20${hours}:${minutes}:${seconds}`;
  } else {
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  }
}
export default function Tab1Screen(props: Tab1ScreenProps) {
  const { user } = useContext(UserContext);
  const { navigation } = props;
  const [anomalyEvents, setAnomalyEvents] = useState<AnomalyEvent[]>([]);
  const [filteredEvents, setFilteredEvents] = useState<AnomalyEvent[]>([]);
  const [searchText, setSearchText] = useState("");
  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchResults, setSearchResults] = useState<AnomalyEvent[]>([]);
  const itemsPerPage = 4;
  const [performSearch, setPerformSearch] = useState(false);

  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [datePickerMode, setDatePickerMode] = useState<"date" | "time">("date");
  const [isStartDatePicker, setIsStartDatePicker] = useState(true);
  const [dateFilter, setDateFilter] = useState(false);

  const onChangeDate = (event, selectedDate) => {
    setShowDatePicker(false);
    if (selectedDate) {
      const currentDate = selectedDate;
      if (datePickerMode === "date") {
        if (isStartDatePicker) {
          setStartDate(currentDate);
        } else {
          setEndDate(currentDate);
        }
        setTimeout(() => {
          setDatePickerMode("time");
          setShowDatePicker(true);
        }, 0);
      } else if (datePickerMode === "time") {
        if (isStartDatePicker) {
          setStartDate((prevDate) => {
            const newDate = new Date(prevDate);
            newDate.setHours(currentDate.getHours(), currentDate.getMinutes());
            return newDate;
          });
        } else {
          setEndDate((prevDate) => {
            const newDate = new Date(prevDate);
            newDate.setHours(currentDate.getHours(), currentDate.getMinutes());
            return newDate;
          });
        }
        setDatePickerMode("date");
      }
    }
  };
  useEffect(() => {
    console.log("startDate 업데이트됨:", formatDateTime(startDate));
    console.log("endDate 업데이트됨:", formatDateTime(endDate));
  }, [startDate, endDate]);
  const renderDatePickerButtons = () => (
    <View style={styles.modalView}>
      <View style={{ flexDirection: "column", alignItems: "center" }}>
        <TouchableOpacity
          onPress={() => {
            setShowDatePicker(true);
            setIsStartDatePicker(true);
            setDatePickerMode("date");
          }}
          style={styles.searchButton}
        >
          <Text style={{ fontFamily: "C24", fontSize: 11 }}>
            시작 날짜 설정
          </Text>
        </TouchableOpacity>
        <Text style={{ fontFamily: "NGB", fontSize: 11 }}>
          {formatDateTime(startDate)}
        </Text>
        <TouchableOpacity
          onPress={() => {
            setShowDatePicker(true);
            setIsStartDatePicker(false);
            setDatePickerMode("date");
          }}
          style={styles.searchButton}
        >
          <Text style={{ fontFamily: "C24", fontSize: 11 }}>
            종료 날짜 설정
          </Text>
        </TouchableOpacity>
        <Text style={{ fontFamily: "NGB", fontSize: 11 }}>
          {formatDateTime(endDate)}
        </Text>
        {showDatePicker && (
          <DateTimePicker
            value={isStartDatePicker ? startDate : endDate}
            mode={datePickerMode}
            is24Hour={true}
            display="default"
            onChange={onChangeDate}
          />
        )}
      </View>
      <TouchableOpacity
        style={styles.modalRegisterButton}
        onPress={() => {
          fetchAnomalyDateEvents(startDate, endDate);
          setDateFilter(false);
        }}
      >
        <Text style={styles.modalButtonText}>적용하기</Text>
      </TouchableOpacity>
    </View>
  );

  const onSearch = () => {
    setPerformSearch((prev) => !prev);
  };

  const fetchAnomalyDateEvents = async (startDate, endDate) => {
    const formattedStartDate =
      startDate !== "" ? formatDateTime(startDate, true) : "";
    const formattedEndDate =
      endDate !== "" ? formatDateTime(endDate, true) : "";

    try {
      const response = await fetch(
        `http://10.28.224.201:30438/api/v0/cctv/loglist_lookup_search?member_id=${user}&start_date=${formattedStartDate}&end_date=${formattedEndDate}`,
        {
          method: "GET",
          headers: { accept: "application/json" },
        },
      );
      console.log("receving data...");
      const data = await response.json();
      console.log(response.ok);
      console.log(
        `http://10.28.224.201:30438/api/v0/cctv/loglist_lookup_search?member_id=${user}&start_date=${formatDateTime(startDate)}&end_date=${formatDateTime(endDate)}`,
      );

      if (response.ok) {
        console.log(data.isSuccess);
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
      fetchAnomalyDateEvents("", "");
    }, [user]),
  );
  useEffect(() => {
    setCurrentPage(1);
    setTotalPages(Math.ceil(searchResults.length / itemsPerPage));
    setSearchText("");
  }, [searchResults]);

  useEffect(() => {
    const filtered = anomalyEvents.filter((event) => {
      return event.cctv_name.toLowerCase().includes(searchText.toLowerCase());
    });

    setSearchResults(filtered);
    setCurrentPage(1);
  }, [anomalyEvents, performSearch]);

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
      <Text style={{ fontSize: 16, fontFamily: "C24", marginBottom: 5 }}>
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
            onPress={() => fetchAnomalyDateEvents("", "")}
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
            placeholder="검색 (CCTV 이름)"
          />
          <TouchableOpacity onPress={onSearch} style={styles.searchButton}>
            <Text style={{ fontFamily: "C24" }}>검색</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => {
              setDateFilter(true);
            }}
            style={styles.searchButton}
          >
            <Text style={{ fontFamily: "C24" }}>필터</Text>
          </TouchableOpacity>
        </View>
        <Overlay
          isVisible={dateFilter}
          overlayStyle={styles.overlayStyle}
          onBackdropPress={() => setDateFilter(false)}
        >
          {renderDatePickerButtons()}
        </Overlay>
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
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#CCCCCC",
    borderRadius: 10,
    padding: 20,
    marginVertical: 8,
    marginHorizontal: 16,
    alignItems: "flex-start",
  },
  timestamp: {
    fontSize: 11,
    color: "#555555",
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
    padding: 20,
    borderWidth: 0,
    borderColor: "black",
  },
  pageItemActive: {
    backgroundColor: "red",
  },
  searchContainer: {
    flexDirection: "row",
    alignItems: "center",
    alignContent: "center",
    margin: 15,
  },
  searchInput: {
    height: 40,
    borderWidth: 1,
    paddingLeft: 8,
    flex: 1,
    borderRadius: 10,
    borderColor: "#CCCCCC",
    marginRight: 8,
  },
  searchButton: {
    padding: 10,
    backgroundColor: "#ddd",
    borderRadius: 10,
    margin: 5,
  },
  refreshButton: {
    marginLeft: 10,
    padding: 10,
    borderRadius: 10,
  },
  refreshButtonText: {
    color: "#000",
    fontSize: 20,
  },
  modalView: {
    margin: 20,
    backgroundColor: "white",
    borderRadius: 20,
    padding: 35,
    alignItems: "stretch",
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
    width: "80%",
  },
  centeredView: {
    justifyContent: "center",
    alignItems: "center",
  },
  modalRegisterButton: {
    marginTop: 20,
    backgroundColor: "grey",
    borderRadius: 20,
    height: 50,
    justifyContent: "center",
    width: "100%",
  },
  modalButtonText: {
    color: "#fff",
    fontSize: 11,
    textAlign: "center",
    fontFamily: "C24",
  },
  overlayStyle: {
    width: "90%",
    backgroundColor: "transparent",
    elevation: 0,
    shadowOpacity: 0,
  },
});
