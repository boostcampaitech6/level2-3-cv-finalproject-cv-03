import React, { useState, useEffect } from "react";
import { Dimensions, View, Text, StyleSheet } from "react-native";
import { Video } from "expo-av";
// import Constants from 'expo-constants';
// import { UserContext } from '../../UserContext'

const Tab2Screen = () => {
  // const { user } = useContext(UserContext)
  const [hlsUrl, setHlsUrl] = useState("");
  const videoWidth = Dimensions.get("window").width;
  const videoHeight = (videoWidth / 11) * 9;

  const hlsuri = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.201:30573/api/v0/streaming/list_lookup?member_id=${105}`,
        // `http://10.28.224.201:30573/api/v0/streaming/list_lookup?member_id=${user}`,
        {
          method: "GET",
          headers: { accept: "application/json" },
        },
      );
      const data = await response.json();
      if (response.ok) {
        setHlsUrl(data.result[0].hls_url);
        console.log("hihihihihi");
        console.log(hlsUrl);
      } else {
        console.error("API 호출에 실패했습니다:", data);
      }
    } catch (error) {
      console.error("API 호출 중 예외가 발생했습니다:", error);
    }
  };
  useEffect(() => {
    hlsuri();
  }, []);

  return (
    <View style={styles.container}>
      <Text>스트리밍 테스트 진행</Text>
      <Video
        source={{
          uri: "https://bitdash-a.akamaihd.net/content/MI201109210084_1/m3u8s/f08e80da-bf1d-4e3d-8899-f0f6155f6efa.m3u8",
        }}
        rate={1.0}
        volume={1.0}
        isMuted={true}
        resizeMode="cover"
        shouldPlay
        style={{ width: videoWidth, height: videoHeight }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    backgroundColor: "#ecf0f1",
  },
  video: {
    alignSelf: "center",
    width: 320,
    height: 200,
  },
  buttons: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
  },
});

export default Tab2Screen;

// const Tab2Screen = () => {
//   const video = React.useRef(null);
//   const [status, setStatus] = React.useState({});
//   return (
//     <View style={styles.container}>
//       <Text>스트리밍 테스트 진행</Text>
//       <Video
//         ref={video}
//         style={styles.video}
//         source={{
//           uri: 'http://10.28.224.201:30573/hls/index.m3u8',
//         }}
//         useNativeControls
//         resizeMode={ResizeMode.CONTAIN}
//         isLooping
//         onPlaybackStatusUpdate={status => setStatus(() => status)}
//       />
//       <View style={styles.buttons}>
//         <Button
//           title={status.isPlaying ? 'Pause' : 'Play'}
//           onPress={() =>
//             status.isPlaying ? video.current.pauseAsync() : video.current.playAsync()
//           }
//         />
//       </View>
//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     justifyContent: 'center',
//     backgroundColor: '#ecf0f1',
//   },
//   video: {
//     alignSelf: 'center',
//     width: 320,
//     height: 200,
//   },
//   buttons: {
//     flexDirection: 'row',
//     justifyContent: 'center',
//     alignItems: 'center',
//   },
// });

// export default Tab2Screen;

// const Tab2Screen = () => {
//   const video = React.useRef(null);
//   const [status, setStatus] = React.useState({});
//   return (
//     <View style={styles.container}>
//       <Text>스트리밍 테스트 진행</Text>
//       <WebView
//         style={styles.container}
//         source={{ uri: 'https://74vod-adaptive.akamaized.net/exp=1710260769~acl=%2F286dea48-7f58-450f-bbad-bbc6b0a611b9%2F%2A~hmac=39c47bc536a1191ba348463f04fb3791e47560ced4e62f7a0ba1ad0aef958c71/286dea48-7f58-450f-bbad-bbc6b0a611b9/sep/video/5ae098b4,7fe452b2,8dbe8e5c,a0b50797,ea5fe2fc/audio/bc6fb25a,dc1a4a6e,e1c6cec8/master.json?base64_init=1&query_string_ranges=1' }}
//       />
//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     marginTop: Constants.statusBarHeight,
//   },
// });

// export default Tab2Screen;
