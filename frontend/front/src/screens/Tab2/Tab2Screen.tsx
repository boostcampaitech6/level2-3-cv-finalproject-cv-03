import React, { useState, useEffect, useRef, useContext } from "react";
import {
  View,
  FlatList,
  TouchableOpacity,
  Modal,
  StyleSheet,
  Dimensions,
  Text,
} from "react-native";
import { Video } from "expo-av";
import { UserContext } from "../../UserContext";

const Tab2Screen = () => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const playbackStatusRef = useRef({});
  const [videos, setVideos] = useState([]);
  const { user } = useContext(UserContext);

  const fetchVideos = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.201:30438/api/v0/streaming/list_lookup?member_id=${user}`,
        {
          method: "GET",
          headers: { accept: "application/json" },
        },
      );
      const data = await response.json();
      if (response.ok) {
        setVideos(data.result);
      } else {
        console.error("API 호출에 실패했습니다:", data);
      }
    } catch (error) {
      console.error("API 호출 중 예외가 발생했습니다:", error);
    }
  };
  useEffect(() => {
    fetchVideos();
  }, []);

  const numColumns = videos.length > 9 ? 4 : videos.length > 6 ? 3 : 2;
  const videoWidth = Dimensions.get("window").width / numColumns;

  const renderVideoItem = ({ item }) => {
    return (
      <View style={{ width: videoWidth }}>
        <TouchableOpacity
          style={[styles.videoItem, { height: videoWidth }]}
          onPress={() => handleVideoSelect(item)}
        >
          <Video
            source={{ uri: item.hls_url }}
            style={styles.video}
            resizeMode="cover"
            isMuted={true}
            shouldPlay={true}
            isLooping
            onPlaybackStatusUpdate={onPlaybackStatusUpdate(item.cctv_id)}
          />
        </TouchableOpacity>
        <Text style={styles.videoTitle}>{item.cctv_name}</Text>
      </View>
    );
  };

  const handleVideoSelect = (video) => {
    if (isFullScreen && selectedVideo?.cctv_id === video.cctv_id) {
      setIsFullScreen(false);
    } else {
      setSelectedVideo(video);
      setIsFullScreen(true);
    }
  };

  useEffect(() => {
    if (!isFullScreen) {
      setSelectedVideo(null);
    }
  }, [isFullScreen]);

  const onPlaybackStatusUpdate = (videoId) => (status) => {
    if (status.isLoaded) {
      playbackStatusRef.current[videoId] = {
        positionMillis: status.positionMillis,
        shouldPlay: status.shouldPlay,
      };
    }
  };

  return (
    <View style={styles.container}>
      <FlatList
        data={videos}
        renderItem={renderVideoItem}
        keyExtractor={(item) => item.cctv_id.toString()}
        numColumns={numColumns}
        key={numColumns}
        contentContainerStyle={styles.gridContentContainer}
      />

      {selectedVideo && isFullScreen && (
        <Modal
          animationType="slide"
          transparent={false}
          visible={isFullScreen}
          onRequestClose={() => setIsFullScreen(false)}
        >
          <TouchableOpacity
            style={styles.fullScreenContainer}
            onPress={() => setIsFullScreen(false)}
          >
            <Video
              source={{ uri: selectedVideo.hls_url }}
              style={styles.fullScreenVideo}
              resizeMode="contain"
              shouldPlay={true}
              positionMillis={
                playbackStatusRef.current[selectedVideo.cctv_id]
                  ?.positionMillis || 0
              }
              isMuted={true}
            />
          </TouchableOpacity>
        </Modal>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gridContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
  },
  gridContentContainer: {
    flexGrow: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  videoItem: {
    justifyContent: "center",
    alignItems: "center",
    alignContent: "center",
  },
  video: {
    width: "100%",
    height: "100%",
  },
  fullScreenContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "black",
  },
  fullScreenVideo: {
    width: Dimensions.get("window").width,
    height: Dimensions.get("window").height,
  },
  videoTitle: {
    fontSize: 16,
    color: "white",
    textAlign: "center",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    width: "100%",
    padding: 5,
    fontFamily: "C24",
    justifyContent: "center",
  },
});

export default Tab2Screen;
