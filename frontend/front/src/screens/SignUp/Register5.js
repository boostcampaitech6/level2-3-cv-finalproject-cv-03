import React, { useState, useEffect } from "react";
import {
  StyleSheet,
  ImageBackground,
  Dimensions,
  StatusBar,
  KeyboardAvoidingView
} from "react-native";
import { Block, Checkbox, Text, theme } from "galio-framework";

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import Icon from 'react-native-vector-icons/FontAwesome';
import { View } from 'react-native';

const { width, height } = Dimensions.get("screen");

const Register5 = (props) => {
  const { navigation, route } = props;
  const { email } = route.params;
  const { password } = route.params;
  // console.log(email, password)
  const [fail, setFail] = useState(true);
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");

  useEffect(() => {
    if (url.length !== 0) {
      setFail(false);
    }
  }, [url]);
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
              <Block flex={0.33} paddingLeft={30} style={{ justifyContent: 'flex-end' }}>
                <Text color="black" size={28} paddingBottom={20} style={styles.subTitle}>
                  회원가입
                </Text>
              </Block>
              <View paddingLeft={30} style={{ flexDirection: 'row' }}>
                <Text color={argonTheme.COLORS.MUTED} size={10} paddingBottom={20} style={styles.text}>
                  1. 약관 동의{" "} > 2. 이메일 인증{" "}>{" "}3. 비밀번호 입력{" "}>{" "}
                </Text>
                <Text color="black" size={10} paddingBottom={20} style={styles.text} >
                  4. URL 등록
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
                      borderless
                      placeholder="CCTV 이름"
                      onChangeText={(text) => {setName(text)}}
                      value={name}
                      iconContent={
                        <Icon
                          size={16}
                          color={argonTheme.COLORS.ICON}
                          name="sticky-note"
                          style={styles.inputIcons}
                        />
                      }
                    />
                  </Block>
                  <Block width={width * 0.8} marginTop={0}>
                    <Input
                      borderless
                      placeholder="CCTV URL"
                      onChangeText={(text) => {setUrl(text)}}
                      value={url}
                      iconContent={
                        <Icon
                          size={16}
                          color={argonTheme.COLORS.ICON}
                          name="link"
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
                    {fail && (
                    <Text style={styles.text2} marginTop={10} marginStart={5} color={argonTheme.COLORS.ERROR}>URL이 잘못되었습니다.</Text>
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
                      onPress={() => navigation.navigate('Register6', { email: email, password: password, cctv_url: url, cctv_name: name })}
                      color={(fail) ? "muted" : "primary"} 
                      style={styles.createButton}
                      disabled={fail} // Button is disabled if either isChecked2 or isChecked3 is not checked
                      textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                    >
                      제출
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
}


const styles = StyleSheet.create({
  registerContainer: {
    width: width * 0.9,
    height: height * 0.875,
    backgroundColor: "#F4F5F7",
    borderRadius: 4,
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1,
    overflow: "hidden"
  },
  socialConnect: {
    backgroundColor: argonTheme.COLORS.WHITE,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: "#8898AA"
  },
  socialButtons: {
    width: 120,
    height: 40,
    backgroundColor: "#fff",
    shadowColor: argonTheme.COLORS.BLACK,
    shadowOffset: {
      width: 0,
      height: 4
    },
    shadowRadius: 8,
    shadowOpacity: 0.1,
    elevation: 1
  },
  socialTextButtons: {
    color: argonTheme.COLORS.PRIMARY,
    fontWeight: "800",
    fontSize: 14
  },
  inputIcons: {
    marginRight: 12
  },
  passwordCheck: {
    paddingLeft: 15,
    paddingTop: 13,
    paddingBottom: 30
  },
  createButton: {
    width: width * 0.5,
    marginTop: 25
  },
  subTitle: {
    fontFamily: 'SG',
    marginTop: 20
  },
  text: {
    fontFamily: 'NGB',
  },
  text2: {
    fontFamily: 'NGB',
    fontSize: 13,
  },
});

export default Register5;
