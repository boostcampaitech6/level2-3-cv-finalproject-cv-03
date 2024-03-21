import React from "react";
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
  ImageBackground,
} from "react-native";
import { Images } from "../../constants";
import { NavigationProp } from "@react-navigation/native";

type RootStackParamList = {
  CctvSettingScreen: undefined;
  Alarm: undefined;
  Profile: undefined;
};

interface Tab2ScreenProps {
  navigation: NavigationProp<
    RootStackParamList,
    "CctvSettingScreen" | "Alarm" | "Profile"
  >;
}

const { width, height } = Dimensions.get("screen");

export default function Tab2Screen(props: Tab2ScreenProps) {
  const { navigation } = props;
  return (
    <ImageBackground
      source={Images.Onboarding}
      style={{ flex: 1, width, height, zIndex: 1 }}
    >
      <View style={styles.container}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate("CctvSettingScreen", {})}
        >
          <Text
            style={{ fontSize: 24, textAlign: "center", fontFamily: "C24" }}
          >
            CCTV
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate("Alarm", {})}
        >
          <Text
            style={{ fontSize: 24, textAlign: "center", fontFamily: "C24" }}
          >
            알림/동영상
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate("Profile", {})}
        >
          <Text
            style={{ fontSize: 24, textAlign: "center", fontFamily: "C24" }}
          >
            개인 정보
          </Text>
        </TouchableOpacity>
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  button: {
    backgroundColor: "#FFFFFF", // White color for the button background
    padding: 20,
    marginVertical: 10,
    borderRadius: 10,
    shadowColor: "rgba(0,0,0, .4)", // Shadow color
    shadowOffset: { height: 1, width: 1 },
    shadowOpacity: 1,
    shadowRadius: 1,
    elevation: 2,
    width: "80%", // Set width to 80% of the container width
  },
  buttonText: {
    fontSize: 18,
    color: "#000", // Black color for the text
    textAlign: "center",
    fontFamily: "NGB",
  },
});
