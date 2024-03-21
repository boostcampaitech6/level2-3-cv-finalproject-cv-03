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

const Register6 = (props) => {
  const { navigation, route } = props;
  const { email } = route.params;
  const { password } = route.params;
  const { cctv_url } = route.params;
  const { cctv_name } = route.params;
  const [name, setName] = useState("default");
  const [member_name, setMember_name] = useState("hong");
  const [fail, setFail] = useState(false);

  const handleRegister = async () => {
    console.log(email, password, name, cctv_url, cctv_name)
    try {
      const response = await fetch(`http://10.28.224.201:30576/api/v0/members/register?email=${encodeURIComponent(email)}&password=${password}&member_name=${member_name}d&store_name=${name}&cctv_url=${cctv_url}&cctv_name=${cctv_name}`, {
        method: 'POST',
        headers: {
          'accept': 'application/json',
        },
        // body: JSON.stringify({ email }),
      });
      // console.log(email)
      const data = await response.json();
      console.log(data);
      if (data.isSuccess) {
        setFail(false)
        navigation.navigate("Login");
      }
      else {
        setFail(true)
    }
    } catch (error) {
      console.error('Network error:', error);
    }
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
              <Block flex={0.33} paddingLeft={30} style={{ justifyContent: 'flex-end' }}>
                <Text color="black" size={28} paddingBottom={20} style={styles.subTitle}>
                  매장 이름 등록
                </Text>
              </Block>
              
              <Block flex center>
                <KeyboardAvoidingView
                  style={{ flex: 1 }}
                  behavior="padding"
                  enabled
                >
                  <Text color="black" size={14} paddingLeft={10} paddingBottom={20} style={styles.text2} marginStart={0}>
                    입력하지 않으시면 기본값으로 설정됩니다.
                  </Text>
                  
                  <Block width={width * 0.8} marginTop={40}>
                    <Input
                      borderless
                      placeholder="매장 이름"
                      onChangeText={(text) => {setName(text)}}
                      value={name}
                      iconContent={
                        <Icon
                          size={16}
                          color={argonTheme.COLORS.ICON}
                          name="shopping-bag"
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
                      // onPress={() => navigation.navigate('Login')}
                      onPress={handleRegister}
                      color={"primary"} 
                      style={styles.createButton}
                      textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                    >
                      제출
                    </Button>
                    {/* <Button 
                      // onPress={() => navigation.navigate('Login')}
                      onPress={handleRegister}
                      color={"primary"} 
                      style={{...styles.createButton, marginTop:7}}
                      textStyle={{ fontSize: 13, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}
                    >
                      건너뛰기
                    </Button> */}

                    {fail && (
                      <Text style={styles.failText}>회원가입에 실패했습니다.</Text>
                    )}
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
    fontFamily: 'SG',
  },
  text2: {
    fontFamily: 'NGB',
    fontSize: 14,
  },
  failText: {
    color: argonTheme.COLORS.ERROR,
    fontFamily: 'NGB',
    fontSize: 13,
  }
});

export default Register6;
