import React, { useState, useContext } from "react";
import {
  StyleSheet,
  Dimensions,
  ScrollView,
  Image,
  ImageBackground,
  Platform,
} from "react-native";
import { KeyboardAwareScrollView } from "react-native-keyboard-aware-scroll-view";
import { Block, Text, theme } from "galio-framework";

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import { HeaderHeight } from "../../constants/utils";
import { UserContext } from "../../UserContext";

const { width, height } = Dimensions.get("screen");

const thumbMeasure = (width - 48 - 32) / 3;

const ProfileEdit = (props) => {
  const { navigation, route } = props;
  const { email } = route.params;
  const { password } = route.params;
  const { store_name } = route.params;
  const [npassword, setNpassword] = useState(password);
  const [nstore_name, setNstore_name] = useState(store_name);

  const { user } = useContext(UserContext);

  const handleEdit = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.201:30576/api/v0/settings/profile_edit?member_id=${user}&email=${email}&password=${npassword}&store_name=${nstore_name}`,
        {
          method: "POST",
          headers: {
            accept: "application/json",
          },
        },
      );
      const data = await response.json();
      if (data.isSuccess) {
        navigation.navigate("Profile");
      } else {
        console.error("Network error2:", error);
      }
    } catch (error) {
      console.error("Network error:", error);
    }
  };

  return (
    <KeyboardAwareScrollView
      resetScrollToCoords={{ x: 0, y: 0 }}
      contentContainerStyle={{ flex: 1 }}
      scrollEnabled={false}
    >
      <Block flex style={styles.profile}>
        <Block flex>
          <ImageBackground
            source={Images.ProfileBackground}
            style={styles.profileContainer}
            imageStyle={styles.profileBackground}
          >
            <ScrollView
              showsVerticalScrollIndicator={false}
              style={{ width, marginTop: "15%" }}
            >
              <Block flex style={styles.profileCard}>
                <Block middle style={styles.avatarContainer}>
                  <Image
                    source={{ uri: Images.ProfilePicture }}
                    style={styles.avatar}
                  />
                </Block>
                <Block style={styles.info}></Block>
                <Block flex>
                  <Block middle style={styles.nameInfo}>
                    <Text
                      bold
                      size={28}
                      color="#32325D"
                      style={styles.textName}
                    >
                      홍길동
                    </Text>
                    <Text
                      size={16}
                      color="#32325D"
                      style={{ ...styles.text, marginTop: 10 }}
                    >
                      사장님
                    </Text>
                  </Block>
                  <Block middle style={{ marginTop: 30, marginBottom: 16 }}>
                    <Block style={styles.divider} />
                  </Block>
                  <Block middle style={{ marginTop: 20 }} row space="between">
                    <Text bold size={16} color="#525F7F" style={styles.text}>
                      이메일
                    </Text>
                    <Text bold size={16} color="#525F7F" style={styles.text2}>
                      {email}
                    </Text>
                  </Block>
                  <Block middle style={{ marginTop: 20 }} row space="between">
                    <Text bold size={16} color="#525F7F" style={styles.text}>
                      PW
                    </Text>
                    {/* <Text bold size={16} color="#525F7F" style={styles.text2}>
                      **********
                    </Text> */}
                    <Input
                      // style={styles.textInput}
                      defaultValue={" "}
                      onChangeText={(text) => setNpassword(text)}
                    />
                  </Block>
                  <Block middle style={{ marginTop: 20 }} row space="between">
                    <Text bold size={16} color="#525F7F" style={styles.text}>
                      매장명
                    </Text>
                    {/* <Text bold size={16} color="#525F7F" style={styles.text2}>
                      {store_name}
                    </Text> */}
                    <Input
                      // style={styles.textInput}
                      defaultValue={store_name}
                      onChangeText={(text) => setNstore_name(text)}
                    />
                  </Block>
                  <Block middle marginTop={50}>
                    <Button
                      onPress={handleEdit}
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
    </KeyboardAwareScrollView>
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
  avatarContainer: {
    position: "relative",
    marginTop: -80,
  },
  avatar: {
    width: 124,
    height: 124,
    borderRadius: 62,
    borderWidth: 0,
  },
  nameInfo: {
    marginTop: 35,
  },
  divider: {
    width: "90%",
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
    fontFamily: "SGB",
    marginStart: 10,
    color: "#172B4D",
    // fontColor: '#172B4D',
  },
  text2: {
    fontFamily: "NGB",
    marginTop: 3,
    marginEnd: 10,
  },
  textName: {
    fontFamily: "SGB",
  },
});

export default ProfileEdit;
