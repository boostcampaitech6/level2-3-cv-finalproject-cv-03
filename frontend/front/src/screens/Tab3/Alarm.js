import React, { useState, useContext } from "react";
import {
  StyleSheet,
  Dimensions,
  ScrollView,
  ImageBackground,
  Platform,
} from "react-native";
import { Block, Text, theme } from "galio-framework";
import { useFocusEffect } from "@react-navigation/native";

import { Button } from "../../components";
import { Images, argonTheme } from "../../constants";
import { HeaderHeight } from "../../constants/utils";
import { UserContext } from "../../UserContext";

const { width, height } = Dimensions.get("screen");

const thumbMeasure = (width - 48 - 32) / 3;

const Alarm = (props) => {
  const { navigation } = props;
  const { user } = useContext(UserContext);
  const [threshold, setThreshold] = useState(0);
  const [save_time_length, setSave_time_length] = useState(0);

  useFocusEffect(
    React.useCallback(() => {
      const fetchData = async () => {
        try {
          const response = await fetch(
            `http://10.28.224.201:30438/api/v0/settings/alarm_lookup?member_id=${user}`,
            {
              method: "GET",
              headers: { accept: "application/json" },
            },
          );
          const data = await response.json();
          // console.log(data);
          if (data.isSuccess) {
            setThreshold(data.result.threshold);
            setSave_time_length(data.result.save_time_length);
          } else {
            console.error("No information:", error);
          }
        } catch (error) {
          console.error("Network error:", error);
        }
      };

      fetchData();
    }, [user]),
  );

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
                  <Text bold size={16} color="#525F7F" style={styles.text2}>
                    {threshold}
                  </Text>
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
                  <Text bold size={16} color="#525F7F" style={styles.text2}>
                    {save_time_length}
                  </Text>
                </Block>
                <Block middle marginTop={30}>
                  <Button
                    onPress={() =>
                      navigation.navigate("AlarmEdit", {
                        threshold: threshold,
                        save_time_length: save_time_length,
                      })
                    }
                    color={"primary"}
                    style={styles.createButton}
                    textStyle={{
                      fontSize: 13,
                      color: argonTheme.COLORS.WHITE,
                      fontFamily: "NGB",
                    }}
                  >
                    수정하기
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
});

export default Alarm;
