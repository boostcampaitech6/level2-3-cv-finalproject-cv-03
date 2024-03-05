import React, { useState, useEffect } from "react";
import {
  StyleSheet,
  ImageBackground,
  Dimensions,
  StatusBar,
  KeyboardAvoidingView
} from "react-native";
import { Block, Checkbox, Text, theme, View } from "galio-framework";

import { Button, Input } from "../../components";
import { Images, argonTheme } from "../../constants";
import Icon from 'react-native-vector-icons/FontAwesome';
import { TouchableOpacity } from 'react-native';


const { width, height } = Dimensions.get("screen");

const Login = (props) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fail, setFail] = useState(false);
  const [fail2, setFail2] = useState(false);


  const { navigation } = props;

  const handleLogin = async () => {
    try {
      const response = await fetch(`http://10.28.224.142:30016/api/v0/members/login?email=${encodeURIComponent(email)}&password=${password}`, {
        method: "POST",
        headers: {
          "accept": "application/json",
        },
        // body: JSON.stringify({ email, password }),
      });

      const data = await response.json();
      console.log(data.isSuccess);

      if (data.isSuccess) {
        const token = data.token;
        
        // 로그인 성공
        console.log("Login successful", data);
        
        // 여기서 필요한 동작을 수행하고 홈 화면으로 이동
        // dispatch(setUser(data.user));
        // AsyncStorage.setItem('authToken', token);
        // setUser(data.user);
        // setUserContext(data.user);
        navigation.navigate("Home");
      } else {
        // 로그인 실패
        console.error("Login failed", data.message);
        setFail(true);
      }
    } catch (error) {
      console.error("Network error:", error);
      setFail2(true);
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
              <Block flex={0.5} paddingLeft={30} style={{ justifyContent: 'flex-end' }}>
                <Text color="black" size={28} paddingBottom={20} style={styles.subTitle}>
                  로그인
                </Text>
              </Block>
              <Block flex center>
                <KeyboardAvoidingView
                  style={{ flex: 1 }}
                  behavior="padding"
                  enabled
                >
                  <Block width={width * 0.8} style={{ marginBottom: 0 }}>
                    <Input
                      borderless
                      placeholder="Email"
                      value={email}
                      onChangeText={(text) => {setEmail(text)}}
                      iconContent={
                        <Icon
                          size={16}
                          color={argonTheme.COLORS.ICON}
                          name="envelope-o"
                          style={styles.inputIcons}
                        />
                      }
                    />
                  </Block>
                  <Block width={width * 0.8}>
                    <Input
                      password
                      borderless
                      placeholder="Password"
                      value={password}
                      onChangeText={(text) => {setPassword(text)}}
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
                  {fail && (
                    <Text style={styles.text}  color={argonTheme.COLORS.ERROR}>회원 정보가 없습니다.</Text>
                  )}
                  {fail2 && (
                    <Text style={styles.text}  color={argonTheme.COLORS.ERROR}>네트워크 에러입니다.</Text>
                  )}

                  <Block middle>
                  <TouchableOpacity onPress={() => navigation.navigate('Register')} >
                    <Text bold size={14} style={{ fontSize: 13, color: argonTheme.COLORS.PLACEHOLDER, fontFamily: 'NGB', marginTop: 20}}>
                      회원가입
                    </Text>
                  </TouchableOpacity>

                    <Text bold size={14} style={{ fontSize: 13, color: argonTheme.COLORS.PLACEHOLDER, fontFamily: 'NGB', marginTop: 20}}>
                      이메일/비밀번호 찾기
                    </Text>

                    <Button color="primary" style={styles.createButton} textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}} onPress={handleLogin}>
                      확인
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
  
});

export default Login;
