import React, { useState } from "react";
import { View, StyleSheet, TouchableOpacity } from "react-native";
import { RouteProp } from "@react-navigation/native";
import { RootStackParamList } from "../../navigation/RootStackNavigator";
import { Text } from "galio-framework";
import { argonTheme } from "../../constants";
import { NavigationProp } from "@react-navigation/native";
import { Overlay } from "react-native-elements";
import { Button } from "../../components";
import { Video, ResizeMode } from "expo-av";
import * as FileSystem from "expo-file-system";
import * as Sharing from "expo-sharing";

type LogDetailScreenRouteProp = RouteProp<
  RootStackParamList,
  "LogDetailScreen"
>;
type LogDetailScreenNavigationProp = NavigationProp<
  RootStackParamList,
  "LogDetailScreen"
>;
type LogDetailScreenProps = {
  route: LogDetailScreenRouteProp;
  navigation: LogDetailScreenNavigationProp;
};

export default function CCTVDetailScreen({
  route,
  navigation,
}: LogDetailScreenProps) {
  const {
    anomaly_create_time,
    anomaly_save_path,
    log_id,
    anomaly_score,
    cctv_name,
  } = route.params;

  const video = React.useRef(null);
  const [fail, setFail] = React.useState(false);
  const [saveVisible, setSaveVisible] = useState(false);
  const [save2Visible, setSave2Visible] = useState(false);
  const [deleteVisible, setDeleteVisible] = useState(false);
  const [delete2Visible, setDelete2Visible] = useState(false);

  const handleDelete = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.201:30435/api/v0/cctv/log_delete?log_id=${log_id}`,
        {
          method: "DELETE",
          headers: {
            accept: "application/json",
          },
          // body: JSON.stringify({ email }),
        },
      );
      // console.log(email)
      const data = await response.json();
      console.log(data);
      if (data.isSuccess) {
        setFail(false);
        navigation.navigate("Tab1Screen");
      } else {
        setFail(true);
      }
    } catch (error) {
      console.error("Network error:", error);
    }
  };

  const handleFeedback = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.201:30435/api/v0/cctv/feedback?log_id=${log_id}&feedback=${1}`,
        {
          method: "PUT",
          headers: {
            accept: "application/json",
          },
          // body: JSON.stringify({ email }),
        },
      );
      // console.log(email)
      const data = await response.json();
      console.log(data);
      if (data.isSuccess) {
        setFail(false);
      } else {
        setFail(true);
      }
    } catch (error) {
      console.error("Network error:", error);
    }
  };

  const downloadVideo = async () => {
    const filename = "anomaly.mp4";
    const fileUri = FileSystem.documentDirectory + filename;
    try {
      const result = await FileSystem.downloadAsync(
        `http://10.28.224.201:30435/api/v0/cctv/${log_id}/video.mp4?video_path=${anomaly_save_path}`,
        // "https://d23dyxeqlo5psv.cloudfront.net/big_buck_bunny.mp4",
        fileUri,
      );
      console.log(result);
      console.log("Download successful:", result);
      await save(result.uri);
      await deleteFile(fileUri);
    } catch (error) {
      console.error("Download error :", error);
    }
  };
  const save = async (uri) => {
    try {
      const isAvailable = await Sharing.isAvailableAsync();
      if (isAvailable) {
        await Sharing.shareAsync(uri);
      } else {
        console.log("Sharing is not available on this device.");
      }
    } catch (error) {
      console.error("Error sharing file:", error);
    }
  };
  const deleteFile = async (fileUri) => {
    try {
      await FileSystem.deleteAsync(fileUri);
      console.log("File deleted successfully");
    } catch (error) {
      console.error("Error deleting file:", error);
    }
  };

  const handleFeedback2 = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.201:30435/api/v0/cctv/feedback?log_id=${log_id}&feedback=${0}`,
        {
          method: "PUT",
          headers: {
            accept: "application/json",
          },
        },
      );
      const data = await response.json();
      console.log(data);
      if (data.isSuccess) {
        setFail(false);
      } else {
        setFail(true);
      }
    } catch (error) {
      console.error("Network error:", error);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>{cctv_name}</Text>
      </View>
      <View style={styles.cctvContainer}>
        <Video
          ref={video}
          style={styles.video}
          source={{
            uri: `http://10.28.224.201:30435/api/v0/cctv/${log_id}/video.mp4?video_path=${anomaly_save_path}`,
          }}
          useNativeControls
          resizeMode={ResizeMode.CONTAIN}
          isLooping
        />
      </View>
      <View style={styles.details}>
        <Text style={styles.detailText}>일시: {anomaly_create_time}</Text>
        <Text style={styles.detailText}>
          이상확률: {(anomaly_score * 100).toFixed(2)}%
        </Text>
        <View style={styles.middle}>
          <TouchableOpacity
            style={styles.feedback_button}
            onPress={() => setSaveVisible(true)}
          >
            <Text style={styles.buttonText}>피드백{"\n"}남기기</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.download_button}
            onPress={() => downloadVideo()}
          >
            <Text style={styles.buttonText}>영상{"\n"}다운로드</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.delete_button}
            onPress={() => setDeleteVisible(true)}
          >
            <Text style={styles.buttonText}>기록{"\n"}삭제하기</Text>
          </TouchableOpacity>
        </View>
        {fail && <Text style={styles.failText}>작업에 실패했습니다.</Text>}
        <Overlay
          isVisible={saveVisible}
          onBackdropPress={() => setSaveVisible(false)}
        >
          <View
            style={{
              alignItems: "center",
              justifyContent: "center",
              padding: 40,
            }}
          >
            <Text style={styles.poptitle}>이 분석이 맞나요?</Text>
            <View
              style={{ flexDirection: "row", justifyContent: "space-between" }}
            >
              <Button
                style={{ marginTop: 20, width: 100 }}
                color="success"
                onPress={() => {
                  handleFeedback();
                  setSaveVisible(false);
                  setSave2Visible(true);
                }}
              >
                <Text
                  style={{
                    fontSize: 14,
                    color: argonTheme.COLORS.WHITE,
                    fontFamily: "NGB",
                  }}
                >
                  예
                </Text>
              </Button>
              <Button
                style={{ marginTop: 20, width: 100 }}
                color="error"
                onPress={() => {
                  handleFeedback2();
                  setSaveVisible(false);
                  setSave2Visible(true);
                }}
              >
                <Text
                  style={{
                    fontSize: 14,
                    color: argonTheme.COLORS.WHITE,
                    fontFamily: "NGB",
                  }}
                >
                  아니오
                </Text>
              </Button>
            </View>
          </View>
        </Overlay>

        <Overlay
          isVisible={save2Visible}
          onBackdropPress={() => setSave2Visible(false)}
        >
          <View
            style={{
              alignItems: "center",
              justifyContent: "center",
              padding: 40,
            }}
          >
            <Text style={styles.poptitle}>피드백이 반영되었습니다.</Text>
          </View>
        </Overlay>

        <Overlay
          isVisible={deleteVisible}
          onBackdropPress={() => setDeleteVisible(false)}
        >
          <View
            style={{
              alignItems: "center",
              justifyContent: "center",
              padding: 40,
            }}
          >
            <Text style={styles.poptitle}>삭제하시겠습니까?</Text>
            <View
              style={{ flexDirection: "row", justifyContent: "space-between" }}
            >
              <Button
                style={{ marginTop: 20, width: 100 }}
                color="success"
                onPress={() => {
                  handleDelete();
                  setDeleteVisible(false);
                  setDelete2Visible(true);
                }}
              >
                <Text
                  style={{
                    fontSize: 14,
                    color: argonTheme.COLORS.WHITE,
                    fontFamily: "NGB",
                  }}
                >
                  예
                </Text>
              </Button>
              <Button
                style={{ marginTop: 20, width: 100 }}
                color="error"
                onPress={() => {
                  setDeleteVisible(false);
                  setDelete2Visible(false);
                }}
              >
                <Text
                  style={{
                    fontSize: 14,
                    color: argonTheme.COLORS.WHITE,
                    fontFamily: "NGB",
                  }}
                >
                  아니오
                </Text>
              </Button>
            </View>
          </View>
        </Overlay>

        <Overlay
          isVisible={delete2Visible}
          onBackdropPress={() => setDelete2Visible(false)}
        >
          <View
            style={{
              alignItems: "center",
              justifyContent: "center",
              padding: 40,
            }}
          >
            <Text style={styles.poptitle}>삭제되었습니다.</Text>
          </View>
        </Overlay>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f0f0f0",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#fff",
  },
  headerText: {
    fontSize: 20,
    fontFamily: "C24",
  },
  cctvContainer: {
    flex: 1,
  },
  video: {
    flex: 1,
    alignSelf: "center",
    width: "100%",
    height: 200,
  },
  details: {
    padding: 20,
    backgroundColor: "#fff",
  },
  detailText: {
    fontSize: 16,
    marginVertical: 4,
    fontFamily: "NGB",
  },
  footer: {
    flexDirection: "row",
    justifyContent: "flex-start",
    paddingVertical: 20,
    backgroundColor: "#fff",
    bottom: 0,
  },
  middle: {
    paddingVertical: 20,
    flexDirection: "row",
    justifyContent: "space-around",
  },
  feedback_button: {
    padding: 10,
    backgroundColor: "#610C9F",
    borderRadius: 5,
    flex: 1,
    marginHorizontal: 10,
  },
  download_button: {
    padding: 10,
    backgroundColor: "#940B92",
    borderRadius: 5,
    flex: 1,
    marginHorizontal: 10,
  },
  delete_button: {
    padding: 10,
    backgroundColor: "#DA0C81",
    borderRadius: 5,
    flex: 1,
    marginHorizontal: 10,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    alignContent: "center",
    textAlign: "center",
    fontFamily: "C24",
  },
  failText: {
    color: argonTheme.COLORS.ERROR,
    fontFamily: "NGB",
    fontSize: 13,
  },
  poptitle: {
    fontFamily: "C24",
    marginBottom: 30,
    fontSize: 20,
  },
});
