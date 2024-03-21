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

const Register2 = (props) => {
  const { navigation, route } = props;
  const { email } = route.params;
  const [isChecked2, setIsChecked2] = useState(false);
  const [isChecked3, setIsChecked3] = useState(false);
  const [auth, setAuth] = useState(false);
  const [sent, setSent] = useState(false); // 인증번호를 전송했습니다.
  const [fail, setFail] = useState(false); // 실패했습니다.
  const [success, setSuccess] = useState(false); // 이메일 인증에 성공
  const [token, setToken] = useState("");

  const handleAuth = async () => {
    try {
      const response = await fetch(`http://10.28.224.201:30576/api/v0/members/confirm_auth?email=${email}&code=${token}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      console.log(email, token)
  
      const data = await response.json();
      console.log(data);

      if (data.isSuccess) {
        setSuccess(true)
        setAuth(true)
        setFail(false)
      }
      else {
        setSuccess(false)
        setAuth(false)
        setFail(true)
    }
    } catch (error) {
      console.error('Network error:', error);
    }
  };

  const handleSendAuth = async () => {
    try {
      const response = await fetch(`http://10.28.224.201:30576/api/v0/members/send_auth?email=${email}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
  
      const data = await response.json();
      console.log(data);

      if (data.isSuccess) {
        setSent(true)
        setFail(false)
      }
      else {
        setSent(false)
        setFail(true)
    }
    } catch (error) {
      console.error('Network error:', error);
    }
  };

  return (
    <Block flex middle>
      <StatusBar hidden />
      <ImageBackground
        source={Images.RegisterBackground}
        style={{ width, height, zIndex: 1 }}
      >
        <Block safe flex middle>
          <Block style={styles.registerContainer}>
            {/* <Block flex={0.25} middle style={styles.socialConnect}>
              <Text color="#8898AA" size={12}>
                Sign up with
              </Text>
              <Block row style={{ marginTop: theme.SIZES.BASE }}>
                <Button style={{ ...styles.socialButtons, marginRight: 30 }}>
                  <Block row>
                    <Icon
                      name="logo-github"
                      family="Ionicon"
                      size={14}
                      color={"black"}
                      style={{ marginTop: 2, marginRight: 5 }}
                    />
                    <Text style={styles.socialTextButtons}>GITHUB</Text>
                  </Block>
                </Button>
                <Button style={styles.socialButtons}>
                  <Block row>
                    <Icon
                      name="logo-google"
                      family="Ionicon"
                      size={14}
                      color={"black"}
                      style={{ marginTop: 2, marginRight: 5 }}
                    />
                    <Text style={styles.socialTextButtons}>GOOGLE</Text>
                  </Block>
                </Button>
              </Block>
            </Block> */}
            <Block flex>
              <Block flex={0.33} paddingLeft={30} style={{ justifyContent: 'flex-end' }}>
                <Text color="black" size={28} paddingBottom={20} style={styles.subTitle}>
                  회원가입
                </Text>
              </Block>
              <View paddingLeft={30} style={{ flexDirection: 'row'}}>
                <Text color={argonTheme.COLORS.MUTED} size={12} paddingBottom={20} style={styles.text}>
                  1. 약관 동의{" "} > {""}
                </Text>
                <Text color="black" size={12} paddingBottom={20} style={styles.text}>
                  2. 이메일 인증{" "}
                </Text>
                <Text color={argonTheme.COLORS.MUTED} size={12} paddingBottom={20} style={styles.text}>
                  > 3. 비밀번호 입력 > 4. URL 등록
                </Text>
              </View>
              <Block flex center>
                <KeyboardAvoidingView
                  style={{ flex: 1 }}
                  behavior="padding"
                  enabled
                >
                  

                    <View marginTop={30} style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Block width={width * 0.55} >
                        {/* <Input
                          borderless
                          placeholder="Email"
                          defaultValue={email}
                          iconContent={
                            <Icon
                              size={16}
                              color={argonTheme.COLORS.ICON}
                              name="envelope-o"
                              style={styles.inputIcons}
                            />
                          }
                        /> */}
                        <Text style={styles.text3}>
                          {email}
                        </Text>
                      </Block>
                      <Button 
                        onPress={handleSendAuth}
                        color="button_color2"
                        style={{...styles.createButton, width: '25%', marginTop:6, marginRight:0}}
                        textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                      >
                        전송
                      </Button>
                    </View>
                    {sent && (
                    <Text style={styles.text2} marginStart={5} color={argonTheme.COLORS.SUCCESS}>전송했습니다.</Text>
                    )}
                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Block width={width * 0.55} >
                        <Input
                          borderless
                          placeholder="인증번호"
                          value={token}
                          onChangeText={(text) => setToken(text)}
                          iconContent={
                            <Icon
                              size={16}
                              color={argonTheme.COLORS.ICON}
                              name="key"
                              style={styles.inputIcons}
                            />
                          }
                        />
                      </Block>
                      <Button 
                        onPress={handleAuth}
                        color="button_color2"
                        style={{...styles.createButton, width: '25%', marginTop:6, marginRight:0 }}
                        textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                      >
                        제출
                      </Button>
                    </View>
                    {success && (
                    <Text style={styles.text2} marginStart={5} color={argonTheme.COLORS.SUCCESS}>인증되었습니다.</Text>
                    )}
                    {fail && (
                    <Text style={styles.text2} marginStart={5} color={argonTheme.COLORS.ERROR}>실패했습니다.</Text>
                    )}
                  
                  <Block middle marginTop={30}>
                    <Button 
                      onPress={() => navigation.navigate('Register4', { email: email })}
                      color={auth ? "primary" : "muted" } 
                      style={styles.createButton}
                      disabled={!auth} // Button is disabled if either isChecked2 or isChecked3 is not checked
                      textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
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
    fontSize: 10,
  },
  text2: {
    fontFamily: 'NGB',
    fontSize: 13,
    marginBottom: 10,
  },
  text3: {
    fontFamily: 'NGB',
    fontSize: 12,
    marginStart: 5,
  },
});

export default Register2;
