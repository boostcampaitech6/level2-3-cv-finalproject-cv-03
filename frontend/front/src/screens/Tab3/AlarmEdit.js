/* eslint-disable */
import React, { useState, useEffect, useContext } from "react";
import {
  StyleSheet,
  Dimensions,
  ScrollView,
  ImageBackground,
  Platform,
} from "react-native";
import { Block, Text, theme } from "galio-framework";
// import DropDownPicker from "react-native-dropdown-picker";
// import { Picker } from "@react-native-picker/picker";

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import { HeaderHeight } from "../../constants/utils";
import { UserContext } from "../../UserContext";
import { Overlay } from "react-native-elements";
import { View } from "react-native";

const { width, height } = Dimensions.get("screen");

const thumbMeasure = (width - 48 - 32) / 3;

const AlarmEdit = (props) => {
  const { navigation, route } = props;
  const { user } = useContext(UserContext);
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [password2Visible, setPassword2Visible] = useState(false);
  const { threshold } = route.params;
  const { save_time_length } = route.params;
  const [nthreshold, setNthreshold] = useState(threshold);
  const [nsave_time_length, setNsave_time_length] = useState(save_time_length);
  const { store_name } = route.params;
  const [nstore_name, setNstore_name] = useState(store_name);

  const handleSave = async () => {
    setPasswordVisible(true);
  };
  const handleEdit = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.201:30438/api/v0/settings/alarm_edit?member_id=${user}&threshold=${nthreshold}&save_time_length=${nsave_time_length}`,
        {
          method: "POST",
          headers: { accept: "application/json" },
        },
      );
      const data = await response.json();
      // console.log(data);
      if (data.isSuccess) {
        navigation.navigate("Alarm");
      } else {
        console.error("No information available");
      }
    } catch (error) {
      console.error("Network error:", error);
    }
  };

  useEffect(() => {
    setNstore_name(store_name);
  }, [store_name]);

  return (
    <Block flex style={styles.profile}>
      <Block flex>
        <ImageBackground
          source={Images.Onboarding}
          style={{ width, height, zIndex: 1 }}
        >
          <ScrollView
            showsVerticalScrollIndicator={false}
            style={{ width, marginTop: "20%" }}
            nestedEnabled={true}
          >
            <Block flex style={styles.profileCard}>
              <Block flex>
                <Text
                  style={{
                    fontFamily: "C24",
                    fontSize: 25,
                    marginStart: 8,
                    marginTop: 10,
                    marginBottom: 5,
                    color: "#172B4D",
                  }}
                >
                  알림
                </Text>
                <Block middle style={{ marginTop: 3, marginBottom: 3 }}>
                  <Block style={styles.divider} />
                </Block>
                <Block middle style={{ marginTop: 20 }} row space="between">
                  <Text bold size={16} color="#525F7F" style={styles.text}>
                    임계값
                  </Text>

                  <Input
                    // style={styles.textInput}
                    defaultValue={threshold.toString()}
                    onChangeText={(text) => setNthreshold(text)}
                  />
                </Block>
                <Text
                  style={{
                    fontFamily: "C24",
                    fontSize: 25,
                    marginStart: 8,
                    marginTop: 10,
                    marginBottom: 5,
                    color: "#172B4D",
                  }}
                >
                  저장 동영상
                </Text>
                <Block middle style={{ marginTop: 3, marginBottom: 3 }}>
                  <Block style={styles.divider} />
                </Block>
                <Block middle style={{ marginTop: 20 }} row space="between">
                  <Text bold size={16} color="#525F7F" style={styles.text}>
                    시간
                  </Text>

                  <Input
                    // style={styles.textInput}
                    defaultValue={save_time_length.toString()}
                    onChangeText={(text) => setNsave_time_length(text)}
                  />
                </Block>

                <Overlay
                  isVisible={passwordVisible}
                  onBackdropPress={() => setPasswordVisible(false)}
                >
                  <View
                    style={{
                      alignItems: "center",
                      justifyContent: "center",
                      padding: 40,
                    }}
                  >
                    <Text style={styles.poptitle}>저장하시겠습니까?</Text>
                    <View
                      style={{
                        flexDirection: "row",
                        justifyContent: "space-between",
                      }}
                    >
                      <Button
                        style={{ marginTop: 20, width: 100 }}
                        color="success"
                        onPress={() => {
                          setPasswordVisible(false);
                          setPassword2Visible(true);
                          handleEdit();
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
                          setPasswordVisible(false);
                          setPassword2Visible(false);
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
                  isVisible={password2Visible}
                  onBackdropPress={() => setPassword2Visible(false)}
                >
                  <View
                    style={{
                      alignItems: "center",
                      justifyContent: "center",
                      padding: 40,
                    }}
                  >
                    <Text style={styles.poptitle}>저장되었습니다.</Text>
                  </View>
                </Overlay>

                <Block middle marginTop={50}>
                  <Button
                    onPress={handleSave}
                    color={"primary"}
                    style={styles.createButton}
                    textStyle={{
                      fontSize: 13,
                      color: argonTheme.COLORS.WHITE,
                      fontFamily: "NGB",
                    }}
                  >
                    저장하기
                  </Button>
                </Block>
              </Block>
            </Block>
          </ScrollView>
        </ImageBackground>
      </Block>
    </Block>
  );
};

const styles = StyleSheet.create({
  profile: {
    marginTop: Platform.OS === "android" ? -HeaderHeight : 0,
    // marginBottom: -HeaderHeight * 2,
    flex: 1,
  },
  profileContainer: {
    width: width,
    height: height,
    padding: 0,
    zIndex: 1,
  },
  profileBackground: {
    width: width,
    height: height / 2,
  },
  profileCard: {
    // position: "relative",
    padding: theme.SIZES.BASE,
    marginHorizontal: theme.SIZES.BASE,
    marginTop: 65,
    borderTopLeftRadius: 6,
    borderTopRightRadius: 6,
    borderRadius: 6,
    backgroundColor: theme.COLORS.WHITE,
    shadowColor: "black",
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 8,
    shadowOpacity: 0.2,
    zIndex: 2,
  },
  info: {
    paddingHorizontal: 40,
  },
  nameInfo: {
    marginTop: 35,
  },
  divider: {
    width: "100%",
    borderWidth: 1,
    borderColor: "#E9ECEF",
  },
  thumb: {
    borderRadius: 4,
    marginVertical: 4,
    alignSelf: "center",
    width: thumbMeasure,
    height: thumbMeasure,
  },
  text: {
    fontFamily: "C24",
    marginStart: 10,
    marginBottom: 30,
    color: "#172B4D",
    marginTop: 20,
  },
  text2: {
    fontFamily: "NGB",
    marginStart: 10,
    marginBottom: 30,
    marginEnd: 10,
  },

  poptitle: {
    fontFamily: "C24",
    marginBottom: 30,
    fontSize: 20,
  },
  textTitle: {
    fontFamily: "C24",
    marginBottom: 0,
    marginStart: 5,
    fontSize: 25,
    color: "#172B4D",
  },
  textInput: {
    width: 100,
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: { width: 0, height: 1 },
    shadowRadius: 2,
    shadowOpacity: 0.05,
    elevation: 2,
    borderRadius: 4,
    borderColor: argonTheme.COLORS.BORDER,
    height: 44,
    backgroundColor: "#FFFFFF",
    margin: 0,
  },
});

export default AlarmEdit;
