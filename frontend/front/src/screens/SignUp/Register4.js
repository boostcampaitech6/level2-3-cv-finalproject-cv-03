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
import { View } from "react-native";

const { width, height } = Dimensions.get("screen");

const Register4 = (props) => {
  const { navigation, route } = props;
  const { email } = route.params; // Access email from navigation parameters
  // console.log(email);
  const [fail, setFail] = useState(true);
  const [fail2, setFail2] = useState(false);
  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");
  useEffect(() => {
    if (password.length !== 0) {
      setFail(false);
    }
    if (password !== password2 && password.length !== 0) {
      setFail2(true);
    } else {
      setFail2(false);
    }
  }, [password, password2]);
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
                  회원가입
                </Text>
              </Block>
              <View paddingLeft={30} style={{ flexDirection: "row" }}>
                <Text
                  color={argonTheme.COLORS.MUTED}
                  size={12}
                  paddingBottom={20}
                  style={styles.text}
                >
                  1. 약관 동의 {">"} 2. 이메일 인증 {">"}{" "}
                </Text>
                <Text
                  color="black"
                  size={12}
                  paddingBottom={20}
                  style={styles.text}
                >
                  3. 비밀번호 입력{" "}
                </Text>
                <Text
                  color={argonTheme.COLORS.MUTED}
                  size={12}
                  paddingBottom={20}
                  style={styles.text}
                >
                  {">"} 4. URL 등록
                </Text>
              </View>
              <Block flex center>
                <KeyboardAvoidingView
                  style={{ flex: 1 }}
                  behavior="padding"
                  enabled
                >
                  <Block width={width * 0.8} marginTop={40}>
                    <Input
                      password
                      borderless
                      placeholder="비밀번호"
                      onChangeText={(text) => {
                        setPassword(text);
                      }}
                      iconContent={
                        <Icon
                          size={16}
                          color={argonTheme.COLORS.ICON}
                          name="key"
                          style={styles.inputIcons}
                        />
                      }
                    />

                    <Input
                      password
                      borderless
                      placeholder="비밀번호 확인"
                      onChangeText={(text) => {
                        setPassword2(text);
                      }}
                      iconContent={
                        <Icon
                          size={16}
                          color={argonTheme.COLORS.ICON}
                          name="key"
                          style={styles.inputIcons}
                        />
                      }
                    />
                    {/* <Block row style={styles.passwordCheck}>
                      <Text size={12} color={argonTheme.COLORS.MUTED}>
                        password strength:
                      </Text>
                      <Text bold size={12} color={argonTheme.COLORS.SUCCESS}>
                        {" "}
                        strong
                      </Text>
                    </Block> */}
                    {fail2 && (
                      <Text
                        style={styles.text2}
                        marginTop={10}
                        marginStart={5}
                        color={argonTheme.COLORS.ERROR}
                      >
                        비밀번호와 비밀번호 확인이 일치하지 않습니다.
                      </Text>
                    )}
                  </Block>
                  {/* <Block row width={width * 0.75}>
                    <Checkbox
                      checkboxStyle={{
                        borderWidth: 3
                      }}
                      color={argonTheme.COLORS.PRIMARY}
                      label="I agree with the"
                    />
                    <Button
                      style={{ width: 100 }}
                      color="transparent"
                      textStyle={{
                        color: argonTheme.COLORS.PRIMARY,
                        fontSize: 14
                      }}
                    >
                      Privacy Policy
                    </Button>
                  </Block> */}
                  <Block middle marginTop={40}>
                    <Button
                      onPress={() =>
                        navigation.navigate("Register5", {
                          email: email,
                          password: password,
                        })
                      }
                      color={fail || fail2 ? "muted" : "primary"}
                      style={styles.createButton}
                      disabled={fail || fail2} // Button is disabled if either isChecked2 or isChecked3 is not checked
                      textStyle={{
                        fontSize: 13,
                        color: argonTheme.COLORS.WHITE,
                        fontFamily: "NGB",
                      }}
                    >
                      다음
                    </Button>
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
    fontFamily: "NGB",
    fontSize: 9,
  },
  text2: {
    fontFamily: "NGB",
    fontSize: 14,
  },
});

export default Register4;
