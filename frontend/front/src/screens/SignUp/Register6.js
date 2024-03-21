import React, { useState, useEffect } from "react";
import {
  StyleSheet,
  ImageBackground,
  Dimensions,
  StatusBar,
  KeyboardAvoidingView,
} from "react-native";
import { Block, Text } from "galio-framework";

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import Icon from "react-native-vector-icons/FontAwesome";

const { width, height } = Dimensions.get("screen");

const Register6 = (props) => {
  const { navigation, route } = props;
  const { email } = route.params;
  const { password } = route.params;
  const { cctv_url } = route.params;
  const { cctv_name } = route.params;
  // eslint-disable-next-line
  const [member_name, setMember_name] = useState("");
  const [fail, setFail] = useState(false);

  useEffect(() => {
    if (member_name.length === 0) {
      setFail(true);
    } else {
      setFail(false);
    }
  }, [member_name]);

  const handleRegister = async () => {
    navigation.navigate("Register7", {
      email: email,
      password: password,
      member_name: member_name,
      cctv_url: cctv_url,
      cctv_name: cctv_name,
    });
  };

  // const handleRegisterSkip = async () => {
  //   setName="기본값";
  //   navigation.navigate("Login");
  // }

  return (
    <Block flex middle>
      <StatusBar hidden />
      <ImageBackground
        source={Images.RegisterBackground}
        style={{ width, height, zIndex: 1 }}
      >
        <Block safe flex middle>
          <Block style={styles.registerContainer}>
            <Block flex>
              <Block
                flex={0.33}
                paddingLeft={30}
                style={{ justifyContent: "flex-end" }}
              >
                <Text
                  color="black"
                  size={28}
                  paddingBottom={20}
                  style={styles.subTitle}
                >
                  사용자 이름 등록
                </Text>
              </Block>

              <Block flex center>
                <KeyboardAvoidingView
                  style={{ flex: 1 }}
                  behavior="padding"
                  enabled
                >
                  <Block width={width * 0.8} marginTop={40}>
                    <Input
                      borderless
                      placeholder="사용자 이름"
                      onChangeText={(text) => {
                        setMember_name(text);
                      }}
                      value={member_name}
                      iconContent={
                        <Icon
                          size={16}
                          color={argonTheme.COLORS.ICON}
                          name="user-circle"
                          style={styles.inputIcons}
                        />
                      }
                    />
                  </Block>

                  <Block middle marginTop={40}>
                    <Button
                      // onPress={() => navigation.navigate('Login')}
                      onPress={handleRegister}
                      color={fail ? "muted" : "primary"}
                      disabled={fail}
                      style={styles.createButton}
                      textStyle={{
                        fontSize: 13,
                        color: argonTheme.COLORS.WHITE,
                        fontFamily: "NGB",
                      }}
                    >
                      다음
                    </Button>

                    {/* {fail && (
                      <Text style={styles.failText}>
                        회원가입에 실패했습니다.
                      </Text>
                    )} */}
                  </Block>
                </KeyboardAvoidingView>
              </Block>
            </Block>
          </Block>
        </Block>
      </ImageBackground>
    </Block>
  );
};

const styles = StyleSheet.create({
  registerContainer: {
    width: width * 0.9,
    height: height * 0.875,
    backgroundColor: "#F4F5F7",
    borderRadius: 4,
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1,
    overflow: "hidden",
  },
  socialConnect: {
    backgroundColor: argonTheme.COLORS.WHITE,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: "#8898AA",
  },
  socialButtons: {
    width: 120,
    height: 40,
    backgroundColor: "#fff",
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1,
  },
  socialTextButtons: {
    color: argonTheme.COLORS.PRIMARY,
    fontWeight: "800",
    fontSize: 14,
  },
  inputIcons: {
    marginRight: 12,
  },
  passwordCheck: {
    paddingLeft: 15,
    paddingTop: 13,
    paddingBottom: 30,
  },
  createButton: {
    width: width * 0.5,
    marginTop: 25,
  },
  subTitle: {
    fontFamily: "SG",
    marginTop: 20,
  },
  text: {
    fontFamily: "SG",
    fontSize: 9,
  },
  text2: {
    fontFamily: "NGB",
    fontSize: 14,
  },
  failText: {
    color: argonTheme.COLORS.ERROR,
    fontFamily: "NGB",
    fontSize: 13,
  },
});

export default Register6;
